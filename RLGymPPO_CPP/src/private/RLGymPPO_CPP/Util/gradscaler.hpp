#pragma once

#include <iterator>
#include <type_traits>

#include <torch/torch.h>
#include <torch/serialize/archive.h>
#include <variant>

// FROM: https://gist.githubusercontent.com/dorpxam/67ad2bc222b2cf567d4a6fc298375e13
// Very awesome implementation by dorpxam, had to make some changes to work with current libtorch:
// - Replace c10::variant/c10::get()/c10:::visit() with std counterparts
// - Fix "'std': ambigous symbol" error by adding "::" before every "std::"
// - Remove serialization operator overloads for "<<" and ">>" (they were causing multi-define errors) 

namespace torch {
	namespace amp {

		template<typename T, typename = void>
		struct is_container : ::std::false_type {};

		template<typename T>
		struct is_container<T, ::std::void_t<decltype(::std::declval<T>().begin()),
			decltype(::std::declval<T>().end()), typename T::value_type>> : ::std::true_type {};

		struct GradScalerOptions
		{
			GradScalerOptions() = default;

			TORCH_ARG(double, init_scale) = pow(2.0, 16);
			TORCH_ARG(double, growth_factor) = 2.0;
			TORCH_ARG(double, backoff_factor) = 0.5;
			TORCH_ARG(int64_t, growth_interval) = 2000;
			TORCH_ARG(bool, enabled) = true;
		};

		class GradScaler
		{
			enum class OptState : int { READY, UNSCALED, STEPPED };

			using PerDeviceTensors = ::std::map<c10::DeviceType, torch::Tensor>;
			using States = ::std::variant<OptState, PerDeviceTensors>;
			using OptimizerStates = ::std::map<::std::string, States>;
			using PerOptimizerStates = ::std::map<::std::uintptr_t, OptimizerStates>;

		public:
			GradScaler(const GradScaler& grad_scaler) = delete;
			GradScaler(GradScaler&& grad_scaler) = default;

			explicit GradScaler(GradScalerOptions const& options = {})
				: _init_scale(options.init_scale())
				, _growth_factor(options.growth_factor())
				, _backoff_factor(options.backoff_factor())
				, _growth_interval(options.growth_interval())
				, _enabled(options.enabled()) {
				if (_enabled && !(torch::cuda::is_available() || torch::hasXLA())) {
					TORCH_WARN("GradScaler is enabled, but CUDA is not available.  Disabling.");
					_enabled = false;
				}
				if (_enabled) {
					TORCH_CHECK(_growth_factor > 1.0, "The growth factor must be > 1.0.");
					TORCH_CHECK(_backoff_factor < 1.0, "The backoff factor must be < 1.0.");
				}
			}

			auto _check_scale_growth_tracker(::std::string const& funcname) -> void {
				static auto fix = "This may indicate your script did not use scaler.scale(loss or outputs) earlier in the iteration.";
				TORCH_CHECK(_scale.defined(), "Attempted " + funcname + " but _scale is None.  " + fix);
				TORCH_CHECK(_growth_tracker.defined(), "Attempted " + funcname + " but _growth_tracker is None.  " + fix);
			}

			auto _lazy_init_scale_growth_tracker(torch::Device const& dev) -> void {
				TORCH_CHECK(!_growth_tracker.defined(), "_growth_tracker initialized before _scale");
				_scale = torch::full({ 1 }, _init_scale, c10::TensorOptions().dtype(torch::kFloat32).device(dev));
				_growth_tracker = torch::full({ 1 }, _init_growth_tracker, c10::TensorOptions().dtype(torch::kInt32).device(dev));
			}

			template <typename T>
			auto scale(T const& values) {
				if constexpr (::std::is_same<T, torch::Tensor>()) {
					if (!_enabled)
						return values;

					assert(values.is_cuda() || values.device().type() == torch::kXLA);

					if (!_scale.defined())
						_lazy_init_scale_growth_tracker(values.device());

					assert(_scale.defined());

					return values * _scale.to(values.device(), true);
				}
				if constexpr (is_container<T>::value) {
					if (!_enabled)
						return values;

					::std::vector<_MultiDeviceReplicator> stash;

					const auto apply_scale = [&](auto&& value, auto&& apply_scale) {
						using Result = ::std::decay_t<decltype(value)>;

						if constexpr (::std::is_same<Result, torch::Tensor>()) {
							assert(value.is_cuda() || value.device().type() == torch::kXLA);

							if (stash.empty()) {
								if (!_scale.defined())
									_lazy_init_scale_growth_tracker(value.device());

								assert(_scale.defined());

								stash.push_back(_MultiDeviceReplicator(_scale));
							}
							return value * stash.front().get(value.device().type());
						}
						if constexpr (is_container<Result>::value) {
							Result result;
							result.reserve(value.size());
							::std::transform(::std::begin(value),
								::std::end(value),
								::std::back_inserter(result),
								[=](auto&& item) {
									return apply_scale(item, apply_scale);
								});
							return result;
						}
					};
					return apply_scale(values, apply_scale);
				}
			}

			auto _unscale_grads_(torch::optim::Optimizer& optimizer, torch::Tensor& inv_scale, torch::Tensor& found_inf, bool allow_fp16) -> PerDeviceTensors {
				auto per_device_inv_scale = _MultiDeviceReplicator(inv_scale);
				auto per_device_found_inf = _MultiDeviceReplicator(found_inf);

				::std::map<c10::DeviceType, ::std::map<c10::ScalarType, ::std::vector<torch::Tensor>>> per_device_and_dtype_grads;

				torch::NoGradGuard nograd;
				for (auto& group : optimizer.param_groups()) {
					for (auto& param : group.params()) {
						assert(instanceof<torch::Tensor>(&param));

						if (!param.grad().defined())
							continue;

						if ((!allow_fp16) && (param.grad().dtype() == torch::kFloat16))
							throw ::std::invalid_argument("Attempting to unscale FP16 gradients.");

						torch::Tensor to_unscale;
						if (param.grad().is_sparse()) {
							if (param.grad().dtype() == torch::kFloat16)
								param.mutable_grad() = param.grad().coalesce();

							to_unscale = param.grad()._values();
						} else
							to_unscale = param.grad();

						per_device_and_dtype_grads[to_unscale.device().type()][to_unscale.dtype().toScalarType()].push_back(to_unscale);
					}

					for (auto&& [device, per_dtype_grads] : per_device_and_dtype_grads) {
						for (auto&& [_, grads] : per_dtype_grads)
							torch::_amp_foreach_non_finite_check_and_unscale_(grads, per_device_found_inf.get(device),
								per_device_inv_scale.get(device));
					}
				}
				return per_device_found_inf._per_device_tensors;
			}

			auto unscale_(torch::optim::Optimizer& optimizer) -> void {
				if (!_enabled)
					return;

				_check_scale_growth_tracker("unscale_");

				auto& optimizer_state = get_per_optimizer_states(id(optimizer));

				if (optimizer_state["stage"] == OptState::UNSCALED)
					throw ::std::runtime_error("unscale_() has already been called on this optimizer since the last update().");
				else
					if (optimizer_state["stage"] == OptState::STEPPED)
						throw ::std::runtime_error("unscale_() is being called after step().");

				assert(_scale.defined());

				auto inv_scale = _scale.to(torch::kDouble).reciprocal().to(at::kFloat);
				auto found_inf = torch::full({ 1 }, 0.0, at::TensorOptions().dtype(at::kFloat).device(_scale.device()));

				optimizer_state["found_inf_per_device"] = _unscale_grads_(optimizer, inv_scale, found_inf, false);
				optimizer_state["stage"] = OptState::UNSCALED;
			}

			auto _maybe_opt_step(torch::optim::Optimizer& optimizer, OptimizerStates& optimizer_state, torch::optim::Optimizer::LossClosure args) -> c10::optional<c10::Scalar> {
				if (optimizer_state.contains("found_inf_per_device")) {
					auto& found_inf_per_device = ::std::get<1>(optimizer_state["found_inf_per_device"]);
					if (!sum(found_inf_per_device)) {
						auto tensor = optimizer.step(args);
						if (tensor.defined())
							return tensor.item();
					}
				}
				return c10::nullopt;
			}

			auto step(torch::optim::Optimizer& optimizer, torch::optim::Optimizer::LossClosure optimizer_args = nullptr) -> c10::optional<c10::Scalar> {
				if (!_enabled) {
					auto res = optimizer.step(optimizer_args);
					if (res.defined())
						return res.item();
					else
						return c10::nullopt;
				}

				if (optimizer_args != nullptr)
					throw ::std::runtime_error("Closure use is not currently supported if GradScaler is enabled.");

				_check_scale_growth_tracker("step");

				auto& optimizer_state = get_per_optimizer_states(id(optimizer));

				if (optimizer_state["stage"] == OptState::STEPPED)
					throw ::std::runtime_error("step() has already been called since the last update().");

				c10::optional<c10::Scalar> retval;

				// if getattr(optimizer, "_step_supports_amp_scaling", False):
				{
					// ------------------------------------------------------------------------
					// TODO: Future Feature
					// ------------------------------------------------------------------------
					// The boolean '_step_supports_amp_scaling' force to use dynamic inspection
					// of class signature (thank's to python reflection) to call step() with 
					// with extra parameters (two tensors : 'grad_scale' & 'found_inf') 
					// ------------------------------------------------------------------------ 
					// Because step() method in torch::optim::Optimizer() class is pure virtual 
					// and fixed to one single parameter (LossClosure functor). It's impossible 
					// currently to mimic the current python's behavior of the extra parameters
					// ------------------------------------------------------------------------ 
				}

				if (optimizer_state["stage"] == OptState::READY)
					unscale_(optimizer);

				auto&& found_inf_per_device = ::std::get<1>(optimizer_state["found_inf_per_device"]);
				TORCH_CHECK(found_inf_per_device.size() > 0, "No inf checks were recorded for this optimizer.");

				retval = _maybe_opt_step(optimizer, optimizer_state, optimizer_args);

				optimizer_state["stage"] = OptState::STEPPED;

				return retval;
			}

			void update(c10::optional<::std::variant<double, torch::Tensor>> const& new_scale = c10::nullopt) {
				if (!_enabled)
					return;

				_check_scale_growth_tracker("update");

				if (new_scale.has_value()) {
					assert(_scale.defined());

					::std::visit([=](auto&& arg) {
						using T = ::std::decay_t<decltype(arg)>;
						if constexpr (::std::is_same_v<T, double>) {
							_scale.fill_(arg);
						} else
							if constexpr (::std::is_same_v<T, torch::Tensor>) {
								static auto reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False.";
								TORCH_CHECK(arg.dtype() == torch::kFloat32, reason);
								TORCH_CHECK(arg.numel() == 1, reason);
								TORCH_CHECK(arg.requires_grad() == false, reason);

								_scale.copy_(arg);
							}
						},
						new_scale.value());
				} else {
					::std::vector<torch::Tensor> found_infs;
					for (auto&& [_, state] : _per_optimizer_states) {
						auto& iterator = ::std::get<1>(state["found_inf_per_device"]);
						for (auto& [_, found_inf] : iterator)
							found_infs.push_back(found_inf.to(_scale.device(), true));
					}
					assert(found_infs.size() > 0, "No inf checks were recorded prior to update.");

					auto& found_inf_combined = found_infs.front();
					if (found_infs.size() > 1)
						for (const auto i : c10::irange(1, found_infs.size()))
							found_inf_combined += found_infs[i];

					torch::_amp_update_scale_(_scale,
						_growth_tracker,
						found_inf_combined,
						_growth_factor,
						_backoff_factor,
						_growth_interval);
				}

				_per_optimizer_states.clear();
			}

			torch::Tensor _get_scale_async() const {
				return _scale;
			}

			double get_scale() const {
				if (_enabled) {
					if (_scale.defined())
						return _scale.item<double>();
					else
						return _init_scale;
				}
				return 1.0;
			}

			double get_growth_factor() const {
				return _growth_factor;
			}

			void set_growth_factor(double new_factor) {
				_growth_factor = new_factor;
			}

			double get_backoff_factor() const {
				return _backoff_factor;
			}

			void set_backoff_factor(double new_factor) {
				_backoff_factor = new_factor;
			}

			int64_t get_growth_interval() const {
				return _growth_interval;
			}

			void set_growth_interval(int64_t new_interval) {
				_growth_interval = new_interval;
			}

			int64_t get_init_growth_tracker() const {
				return _init_growth_tracker;
			}

			void set_init_growth_tracker(int64_t new_value) {
				_init_growth_tracker = new_value;
			}

			int64_t _get_growth_tracker() const {
				if (_enabled) {
					if (_growth_tracker.defined())
						_growth_tracker.item<int64_t>();
					else
						return _init_growth_tracker;
				}
				return 0;
			}

			bool is_enabled() const {
				return _enabled;
			}

		private:
			template <typename Type>
			inline Type read(torch::serialize::InputArchive& archive, ::std::string const& name) {
				c10::IValue ivalue;
				bool exists = archive.try_read(name, ivalue);
				if (exists)
					return ivalue.to<Type>();
				else
					return Type();
			}

		public:
			void save(torch::serialize::OutputArchive& archive) const {
				if (_enabled) {
					TORCH_CHECK(_per_optimizer_states.empty(),
						"A GradScaler instance may only be saved at the beginning " \
						"of an iteration, or at the end after scaler.update().");

					serialize::OutputArchive state(archive.compilation_unit());
					{
						state.write("scale", get_scale());
						state.write("growth_factor", _growth_factor);
						state.write("backoff_factor", _backoff_factor);
						state.write("growth_interval", _growth_interval);
						state.write("_growth_tracker", _get_growth_tracker());
					}
					archive.write("gradscaler", state);
				}
			}

			void load(torch::serialize::InputArchive& archive) {
				if (!_enabled)
					return;

				if (archive.keys().empty())
					throw ::std::runtime_error("The source state dict is empty, possibly because it was saved " \
						"from a disabled instance of GradScaler.");

				serialize::InputArchive state;
				if (archive.try_read("gradscaler", state)) {
					_init_scale = read<double>(state, "scale");
					if (_scale.defined())
						_scale.fill_(_init_scale);
					_growth_factor = read<double>(state, "growth_factor");
					_backoff_factor = read<double>(state, "backoff_factor");
					_growth_interval = read<int64_t>(state, "growth_interval");
					_init_growth_tracker = read<int64_t>(state, "_growth_tracker");
					if (_growth_tracker.defined())
						_growth_tracker.fill_(_init_growth_tracker);
				}
			}

		private:
			double _init_scale;
			double _backoff_factor;
			double _growth_factor;
			int64_t _growth_interval;

		private:
			at::Tensor _scale;
			at::Tensor _growth_tracker;
			int64_t _init_growth_tracker{ 0 };

		protected:
			bool _enabled;

		private:
			template <typename Type>
			inline ::std::uintptr_t id(Type const& type) {
				return reinterpret_cast<::std::uintptr_t>(::std::addressof(type));
			}

			template <typename Base, typename Type>
			inline bool instanceof(const Type* ptr) {
				return dynamic_cast<const Base*>(ptr) != nullptr;
			}

		private:
			PerOptimizerStates _per_optimizer_states;

			inline auto _refresh_per_optimizer_state() -> OptimizerStates {
				return
				{
					{ "stage", OptState::READY },
					{ "found_inf_per_device", {} }
				};
			}

			inline OptimizerStates& get_per_optimizer_states(::std::uintptr_t optimizer_id) {
				if (_per_optimizer_states.contains(optimizer_id) == false)
					_per_optimizer_states[optimizer_id] = _refresh_per_optimizer_state();

				return _per_optimizer_states.at(optimizer_id);
			}

			friend bool operator==(States const& lhs, OptState const& rhs) {
				return (lhs.index() > 0) ? false : ::std::get<0>(lhs) == rhs;
			}

		private:
			class _MultiDeviceReplicator
			{
			public:
				_MultiDeviceReplicator(torch::Tensor& master_tensor)
					: master(master_tensor) {
					assert(master_tensor.is_cuda() || master_tensor.device().type() == torch::DeviceType::XLA);
				}

				inline torch::Tensor& get(c10::DeviceType device) {
					if (!_per_device_tensors.contains(device))
						_per_device_tensors[device] = master.to(device, true, true);
					return _per_device_tensors[device];
				}

				torch::Tensor& master;
				PerDeviceTensors _per_device_tensors;
			};

			template <typename Type = double>
			inline auto sum(PerDeviceTensors const& per_device) {
				Type sum = Type(0);
				for (auto&& [_, v] : per_device)
					sum += v.item<Type>();
				return sum;
			}
		};

#if 0
		serialize::OutputArchive& operator<< (serialize::OutputArchive& archive, const GradScaler& scaler) {
			scaler.save(archive); return archive;
		}

		serialize::InputArchive& operator>>(serialize::InputArchive& archive, GradScaler& scaler) {
			scaler.load(archive); return archive;
		}
#endif

	} // namespace amp
} // namespace torch