web: gunicorn StockApp.wsgi --preload --log-file - --max-requests 1200
worker: ps:scale web=1
worker: ps:resize worker=standard-2x
