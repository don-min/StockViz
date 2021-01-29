web: gunicorn StockApp.wsgi --preload --log-file - --max-requests 1200
heroku config:set WEB_CONCURRENCY=1
