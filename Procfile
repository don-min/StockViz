web: gunicorn StockApp.wsgi --preload --log-file - --max-requests 1200
heroku ps:scale web=1
heroku ps:resize worker=standard-2x
