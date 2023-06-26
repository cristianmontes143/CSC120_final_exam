from flask import Flask, render_template
import pickle

def create_web():
    web = Flask(__name__)
    web.config['SECRET_KEY'] = 'cristian gwapo'
    
    from.views import views
    from.auth import auth
    
    web.register_blueprint(views, url_prefix='/')
    web.register_blueprint(auth, url_prefix='/')
    return web




