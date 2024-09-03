from flask import Flask, make_response, render_template
app = Flask(__name__)
@app.route('/')
def index():
	response = make_response(render_template("index.html"))
	return response
if __name__ == '__main__':
	app.run()
