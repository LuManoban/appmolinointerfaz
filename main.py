#Libraries
from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
from ultralytics import YOLO
import math

#show images
def images(img, img2, img3):
    img = img
    img2 = img2
    img3 = img3

    if img is not None:
        #Img detect
        img = np.array(img, dtype='uint8')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        img_ = ImageTk.PhotoImage(image=img)
        lblimg.configure(image=img_)
        lblimg.image = img_
    else:
        lblimg.configure(image=transparent_photo)
        lblimg.image = transparent_photo

    if img2 is not None:
        # Img txt
        img2 = np.array(img2, dtype='uint8')
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img2)

        img_2 = ImageTk.PhotoImage(image=img2)
        lblimg2.configure(image=img_2)
        lblimg2.image = img_2
    else:
        lblimg2.configure(image=transparent_photo)
        lblimg2.image = transparent_photo

    if img3 is not None:
        # Img txt3
        img3 = np.array(img3, dtype='uint8')
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        img3 = Image.fromarray(img3)

        img_3 = ImageTk.PhotoImage(image=img3)
        lblimg3.configure(image=img_3)
        lblimg3.image = img_3
    else:
        lblimg3.configure(image=transparent_photo)
        lblimg3.image = transparent_photo


#Scanning Function
def Scanning():
    global lblimg, lblimg2, lblimg3
    detect_molinosag, detect_pernos, detect_liners = False, False, False


    #Interfaz
    lblimg = Label(pantalla)
    lblimg.place(x=75, y=260)
    lblimg2 = Label(pantalla)
    lblimg2.place(x=995,y=150)
    lblimg3 = Label(pantalla)
    lblimg3.place(x=995, y=400)



    #Read videocapture
    if cap is not None:
        ret, frame = cap.read()
        frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret == True:

            results = model(frame, stream=True, verbose=False)

            for res in results:
                # Box
                boxes = res.boxes
                masks = res.masks
                #print(res)
                for box in boxes:
                    # Boonding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    #Error
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 < 0: x2 = 0
                    if y2 < 0: y2 = 0

                    #Clase
                    cls= int(box.cls[0])

                    #Confidence
                    conf = box.conf[0]
                    CONFIDENCE_THRESHOLD = 0.90
                    print(f"Clase: {cls}, Confianza: {conf}")

                for idx, box in enumerate(boxes):
                    if conf > CONFIDENCE_THRESHOLD:
                        if cls == 0:
                            detect_molinosag = True
                            #Draw Rectangulo
                            cv2.rectangle(frame_show, (x1,y1), (x2,y2), (255,0,0),2)

                            #text
                            text = f'{clsName[cls]} {int(conf * 100)}%'
                            sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,1 ,2)
                            dim = sizetext[0]
                            baseline = sizetext[1]
                            #Rect
                            cv2.rectangle(frame_show, (x1,y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0,0,0), cv2.FILLED)
                            cv2.putText(frame_show, text, (x1,y1 -5), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2)

                            #Image
                            images(img_general, None, None)

                    #else:
                        #detect_molinosag = False
                        #images(None, None, None)

                    if conf > CONFIDENCE_THRESHOLD:
                        if cls == 1:
                            detect_pernos = True
                            # Draw Rectangulo
                            cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 128, 0), 2)

                            # text
                            text = f'{clsName[cls]} {int(conf * 100)}%'
                            sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                            dim = sizetext[0]
                            baseline = sizetext[1]
                            # Rect
                            cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline),
                                          (0, 0, 0), cv2.FILLED)
                            cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 2)

                            #Image
                            images(None,img_pernos,None)

                    if conf > CONFIDENCE_THRESHOLD:
                        if cls == 2:
                            detect_liners = True
                            # Draw Rectangulo
                            cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # text
                            text = f'{clsName[cls]} {int(conf * 100)}%'
                            sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                            dim = sizetext[0]
                            baseline = sizetext[1]
                            # Rect
                            cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline),
                                          (0, 0, 0), cv2.FILLED)
                            cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            # Image
                            images(None, None, img_liners)

            molinosag_img = img_general if detect_molinosag else None
            pernos_img = img_pernos if detect_pernos else None
            liners_img = img_liners if detect_liners else None

            images(molinosag_img, pernos_img, liners_img)

            #Resize
            frame_show = imutils.resize(frame_show, width=640)

            #Convwetir Video
            im = Image.fromarray(frame_show)
            img = ImageTk.PhotoImage(image=im)

            #Mostrar
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, Scanning)
        else:
            cap.release()

#main
def ventana_principal():
    global  img_general, img_liners, img_pernos,transparent_photo
    global model , clsName,  cap, lblVideo , pantalla

    # Ventana principal
    pantalla = Tk()
    pantalla.title("MOLINO SAG")
    pantalla.geometry("1280x720")

    transparent_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))  # 1x1 pixel transparent image
    transparent_photo = ImageTk.PhotoImage(image=transparent_img)

    #background
    imagenF = PhotoImage(file="setUp/Ventana.png")
    background = Label(image=imagenF)
    background.place(x=0, y=0, relwidth=1, relheight=1)

    #Model
    model = YOLO('Modelos/molinosag.pt')

    #Clases
    clsName = ['MOLINOSAG', 'PERNOS' , 'LINERS']

    #Img
    img_liners = cv2.imread('setUp/liners.PNG')
    img_general = cv2.imread('setUp/general.png')
    img_pernos = cv2.imread('setUp/pernos.png')

    # Label Video
    lblVideo = Label(pantalla)
    lblVideo.place(x=330, y=150)

    #Cam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4,480)

    #Scanning
    Scanning()

    #Cam
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4,480)


    #Loop
    pantalla.mainloop()

if __name__ == '__main__':
    ventana_principal()


