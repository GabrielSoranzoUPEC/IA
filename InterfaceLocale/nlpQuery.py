import pickle
import socket

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server for NLP
PORT2 = 65433 # The port used by the server for NLP find container

s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def connect():
    print("Connexion aux modèles NLP")
    s.connect((HOST, PORT))

def close():
    s.close()

def findAction(sentence):
    message=("classifier",sentence)
    data=pickle.dumps(message)
    s.send(data)  # send message
    data = s.recv(1024)  # receive response
    reponse=pickle.loads(data)
    return reponse

def findTarget(text,action):
    message=("qa",(text,action))
    data=pickle.dumps(message)
    s.send(data)  # send message
    data = s.recv(1024)  # receive response
    reponse=pickle.loads(data)
    return reponse

