import pickle
import socket
from nlp import *

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while 1==1:
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    print("Deconnexion du client")
                    break
                category,question=pickle.loads(data)
                print("from connected user, type: ", category,", text(s): ",question)
                if category=="classifier":
                    reponse=findAction(question)
                elif category=="qa":
                    reponse=findTarget(*question)
                data=pickle.dumps(reponse)
                conn.send(data)  # send data to the client

