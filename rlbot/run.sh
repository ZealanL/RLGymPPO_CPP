#!/bin/bash
cd "$(dirname "$0")"

python3 -m pip install -r requirements.txt --upgrade

python3 -c "from rlbot import runner; runner.main();"