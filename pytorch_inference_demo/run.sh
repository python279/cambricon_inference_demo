#!/bin/sh

pip3 install gunicorn opencv-python

if [ ${USE_MLU} ]; then
	python Service.py
else
	gunicorn --timeout 600 --graceful-timeout 600 --log-level INFO -w 4 -b 0.0.0.0:5005 Service:app
fi
