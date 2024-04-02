a tts llm word server in python with word segmentation and time stamps


Before starting the server, initialize and activate a Python virtual environment, and install the necessary requirements.
```bash
$ python3 -m venv env 
$ source env/bin/activate 
$ pip install -r requirements.txt
```

Start the API server.
```bash
$ ./start-app.sh
```

Sent a test POST request.
```bash
$ curl -X POST -H "Content-Type: application/json" -d '{"prompt":"hi there, my name is human."}' http://localhost:8000/v1/tts
```
