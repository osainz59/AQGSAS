# AQGSAS
### Automatic Question Generation and Short Answer Scoring system

![Image of application design](img/application_design.png)


## Question Generation model

The model used on this application is the checkpoint provided by [UniLM](https://github.com/microsoft/unilm) that you can download from here: [here](https://drive.google.com/open?id=11E3Ij-ctbRUTIQjueresZpoVzLMPlVUZ).

Once you downloaded the checkpoint create the `pretrained_models/` directory and save it there.

## How to run

### Linux
```bash
$ export FLASK_APP=server.py
$ flask run --host=YOUR_LOCAL_IP --port=PORT
```

### Windows
```powershell
$env:FLASK_APP = "server.py"
flask run --host=YOUR_LOCAL_IP --port=PORT
```