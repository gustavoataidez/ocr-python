import face_recognition as fr

def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)
    if len(rostos) > 0:
        return True, rostos
    return False, []

def get_rostos():
    rostos_conhecidos = []
    nomes_dos_rostos = []

    roger1 = reconhece_face("./img/robertfoto.jpg")
    if roger1[0]:
        rostos_conhecidos.append(roger1[1][0])
        nomes_dos_rostos.append("Robert")

    gustavo1 = reconhece_face("./img/gustavo1.jpeg")
    if gustavo1[0]:
        rostos_conhecidos.append(gustavo1[1][0])
        nomes_dos_rostos.append("Gustavo")

    return rostos_conhecidos, nomes_dos_rostos
