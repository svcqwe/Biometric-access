import customtkinter as ctk
import tkinter as tk
import numpy as np
from PIL import Image
import cv2
from PIL import ImageGrab, Image
from realutils.face.insightface import isf_analysis_faces
from psycopg2 import connect
import time
from CNN_extractor import preprocess_signature, SignatureExtractor, get_embedding, compare_embeddings
import torch

ctk.set_appearance_mode("Dark")  # Темная тема
ctk.set_default_color_theme("blue")  # Синие акценты

device = "cpu"

model = SignatureExtractor(embedding_dim=128).to(device)
model.load_state_dict(torch.load("D:/ho-ho/Pet-projects/Доступ в помещение/Разработка/signature_extractor_weights.pth", map_location=device))

def show_notification(title, message, parent=None):
    """
    Универсальная функция для вывода уведомлений пользователю.
    Заменяет стандартные консольные print().
    """
    if parent is None:
        parent = win_main
    notif = ctk.CTkToplevel(parent)
    notif.title(title)
    notif.geometry("450x220")
    notif.resizable(False, False)
    notif.attributes("-topmost", True)
    notif.grab_set()  # Блокируем взаимодействие с другими окнами, пока открыто уведомление
    
    lbl_title = ctk.CTkLabel(notif, text=title, font=("Roboto", 18, "bold"))
    lbl_title.pack(pady=(20, 10))
    
    lbl_msg = ctk.CTkLabel(notif, text=message, font=("Roboto", 14), wraplength=400)
    lbl_msg.pack(expand=True, padx=20, pady=5)
    
    btn_ok = ctk.CTkButton(notif, text="ОК", command=notif.destroy, width=120, corner_radius=8)
    btn_ok.pack(pady=(0, 20))


def open_win_login():
    win_login = ctk.CTkToplevel(win_main)
    win_login.title("Вход в систему")
    win_login.geometry('1200x700')
    
    # Главный контейнер для центрирования и красоты
    login_frame = ctk.CTkFrame(win_login, fg_color="transparent")
    login_frame.pack(expand=True, fill="both", padx=20, pady=20)
    
    header_lbl = ctk.CTkLabel(login_frame, text="Авторизация", font=("Roboto", 28, "bold"))
    header_lbl.pack(pady=(20, 40))

    camera = cv2.VideoCapture(0)  # ВКЛЮЧАЕМ КАМЕРУ
    if not camera.isOpened():
        raise Exception("Не удалось открыть камеру")
    
    def bytes_to_embedding(data):
        return np.frombuffer(
            data,
            dtype=np.float32
        )
    
    def take_photo():
        success, frame = camera.read()
        if not success:
            raise Exception("Не удалось получить кадр")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        faces = isf_analysis_faces(frame)
        if(len(faces) == 0):
            show_notification("Внимание", "Лицо не найдено! Закройте окно и пройдите процесс регистрации занаво.", win_login)
            return 
        if(len(faces) > 1):
            show_notification("Внимание", "В кадре должны быть только ВЫ! Закройте окно и пройдите процесс регистрации занаво.", win_login)
            return
        
        new_embedding = faces[0].embedding

        try:
            conn = connect(
                 host="localhost",
                 dbname="Access_db",
                 user="postgres",
                 password="09710972",
                 port="5432"
            )

            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        id_face,
                        embedding_1,
                        embedding_2,
                        embedding_3,
                        embedding_4,
                        embedding_5
                    FROM faces
                """)
                
                faces_from_db = cursor.fetchall()
                best_match = None
                best_similarity = 0
                
                for row in faces_from_db:
                    id_face = row[0]
                    db_embeddings = []
                    for emb_bytes in row[1:]:
                        emb = bytes_to_embedding(emb_bytes)
                        db_embeddings.append(emb)

                    for db_embedding in db_embeddings:
                        similarity = np.dot(
                            new_embedding,
                            db_embedding
                        ) / (
                            np.linalg.norm(new_embedding) *
                            np.linalg.norm(db_embedding)
                        )


                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = id_face

                if(best_similarity > 0.7):
                    with conn.cursor() as cursor:
                        cursor.execute("""SELECT fio FROM personal_info WHERE id_face = %s""", (best_match,))
                        fio = cursor.fetchone()[0]
                    if fio is not None:
                        btn_take_photo.pack_forget()
                        header_lbl.configure(text=f"Здравствуйте, {fio}!")
                        show_notification("Успешно", f"Здравствуйте, {fio}!\nПожалуйста, распишитесь на экране ниже.", win_login)

                        # Оборачиваем Canvas в красивую рамку (CTkFrame)
                        canvas_frame = ctk.CTkFrame(login_frame, corner_radius=10, fg_color="gray80")
                        canvas_frame.pack(pady=20)
                        
                        canvas = tk.Canvas(canvas_frame, width=500, height=500, bg="white", highlightthickness=0, cursor="pencil")
                        canvas.pack(padx=2, pady=2)

                        start_time = None
                        end_time = None
                        last_x = None
                        last_y = None

                        def start_draw(event):
                            global start_time
                            start_time = time.perf_counter()
                            nonlocal last_x, last_y
                            last_x = event.x
                            last_y = event.y
                            
                        def stop_draw(event):
                            global start_time, end_time
                            end_time = time.perf_counter()
                            nonlocal last_x, last_y
                            last_x = None
                            last_y = None

                            scale = ctk.ScalingTracker.get_window_scaling(win_login)

                            # Переводим логические координаты Canvas в реальные пиксели экрана
                            x = int(canvas.winfo_rootx() * scale)
                            y = int(canvas.winfo_rooty() * scale)
                            x1 = int((canvas.winfo_rootx() + canvas.winfo_width()) * scale)
                            y1 = int((canvas.winfo_rooty() + canvas.winfo_height()) * scale)
                            # Делаем точный снимок
                            signature_login = ImageGrab.grab(bbox=(x, y, x1, y1)).convert("L")

                            canvas.delete("all")

                            try:
                                conn = connect(
                                     host="localhost",
                                     dbname="Access_db",
                                     user="postgres",
                                     password="09710972",
                                     port="5432"
                                )

                                with conn.cursor() as cursor:
                                    cursor.execute("""SELECT id_signature FROM personal_info WHERE id_face = %s""", (best_match,))
                                    id_signature = cursor.fetchone()[0]

                                    cursor.execute("""SELECT (time_sign_1 + time_sign_2 + time_sign_3 + time_sign_4 + time_sign_5) / 5.0 FROM signatures WHERE id_signature = %s""", (id_signature,))
                                    time_sign_mean = cursor.fetchone()[0]

                                    cursor.execute("""SELECT embedding_1, embedding_2, embedding_3, embedding_4, embedding_5 FROM signatures WHERE id_signature = %s""", (id_signature,))
                                    sign_embeddings = cursor.fetchone()
                                    
                                try:
                                    cos_dist_mean = 0
                                    euc_dist_mean = 0
                                    for db_sign in sign_embeddings:
                                        emb1 = bytes_to_embedding(db_sign)
                                        processed_img = preprocess_signature(signature_login)
                                        emb2 = get_embedding(model, processed_img, device)
                                        cos_dist, euc_dist = compare_embeddings(emb1, emb2)

                                        cos_dist_mean += cos_dist
                                        euc_dist_mean += euc_dist

                                        #print("\n=== Результаты сравнения подписей===")
                                        #print(f"Косинусное расстояние: {cos_dist:.4f} (ближе к 0 -> подписи похожи)")
                                        #print(f"Евклидово расстояние:  {euc_dist:.4f} (ближе к 0 -> подписи похожи)")
                                    print(f"\nСреднее косинусное растояение: {cos_dist_mean / 5}")
                                    print(f"Среднее Евклидово расстояние: {euc_dist_mean / 5}")

                                    score = 0.65 * cos_dist_mean + 0.15 * euc_dist_mean + 0.05 * np.fabs(time_sign_mean - (end_time - start_time))
                                    print(f"Итоговая метрика: {score}")
                            
                                except Exception as e:
                                    show_notification("Ошибка", f"Ошибка выполнения: {e}", win_login)
                                
                                if(score < 0.4):
                                    win_login.destroy()
                                    show_notification("Статус входа", f"Вы успешно вошли!", win_main)
                                else:
                                    win_login.destroy()
                                    show_notification("Статус входа", f"Ошибка! Похоже - это не ваша подпись. Попробуйте занаво!", win_main)
                                    
                            
                            except Exception as e:
                                if conn:
                                    conn.rollback()  # отменяет все изменения в базе данных, которые были сделаны в текущей транзакции, но еще не были подтверждены через conn.commit()
                                show_notification("Ошибка БД", f"Err: {e}", win_login)

                        def draw(event):
                            nonlocal last_x, last_y
                            if last_x is None or last_y is None:
                                last_x = event.x
                                last_y = event.y
                                return
                            canvas.create_line(last_x, last_y, event.x, event.y, width=5, fill="black", capstyle=tk.ROUND)
                            last_x = event.x
                            last_y = event.y

                        canvas.bind("<Button-1>", start_draw)
                        canvas.bind("<ButtonRelease-1>", stop_draw)
                        canvas.bind("<B1-Motion>", draw)

                    else:
                        show_notification("Ошибка", "Пользователь не найден в БД", win_login)
                else:
                    show_notification("Внимание", "Пользователь не распознан, сделайте фото еще раз", win_login)

        except Exception as e:
            if conn:
                conn.rollback()  # отменяет все изменения в базе данных, которые были сделаны в текущей транзакции, но еще не были подтверждены через conn.commit()
            show_notification("Ошибка", f"Err: {e}", win_login)
        finally:
            if conn:
                conn.close()

    btn_take_photo = ctk.CTkButton(login_frame, width=220, height=50, text='Сфотографировать', font=("Roboto", 16, "bold"), corner_radius=10, command=take_photo)
    btn_take_photo.pack(pady=10)

def open_win_register():
    win_register = ctk.CTkToplevel(win_main)
    win_register.title("Регистрация нового пользователя")
    win_register.geometry('1200x800')

    # Контейнер для упорядочивания дизайна
    register_frame = ctk.CTkFrame(win_register, fg_color="transparent")
    register_frame.pack(expand=True, fill="both", padx=30, pady=30)

    label_enter_fio = ctk.CTkLabel(register_frame, text="Введите ФИО:", font=("Roboto", 22, "bold"))
    label_enter_fio.pack(pady=(10, 5))
    textbox_fio = ctk.CTkEntry(register_frame, width=350, height=45, font=("Roboto", 16), placeholder_text="Иванов Иван Иванович", corner_radius=8)
    textbox_fio.pack(pady=(0, 20))

    label_enter_sign = ctk.CTkLabel(register_frame, text="Распишитесь (5 раз):", font=("Roboto", 22, "bold"))
    label_enter_sign.pack(pady=(10, 5))
    
    canvas_container = ctk.CTkFrame(register_frame, corner_radius=10, fg_color="gray80")
    canvas_container.pack(pady=10)
    canvas = tk.Canvas(canvas_container, width=500, height=500, bg="white", highlightthickness=0, cursor="pencil")
    canvas.pack(padx=2, pady=2)

    start_time = None
    end_time = None

    last_x = None
    last_y = None

    reference_signatures = []
    reference_signatures_bytes = []
    time_for_signatures = []

    def start_draw(event):
        global start_time
        start_time = time.perf_counter()
        nonlocal last_x, last_y
        last_x = event.x
        last_y = event.y

    def stop_draw(event):
        global start_time, end_time
        end_time = time.perf_counter()
        nonlocal last_x, last_y
        last_x = None
        last_y = None


        scale = ctk.ScalingTracker.get_window_scaling(win_register)

        # Переводим логические координаты Canvas в реальные пиксели экрана
        x = int(canvas.winfo_rootx() * scale)
        y = int(canvas.winfo_rooty() * scale)
        x1 = int((canvas.winfo_rootx() + canvas.winfo_width()) * scale)
        y1 = int((canvas.winfo_rooty() + canvas.winfo_height()) * scale)

        # Делаем точный снимок
        img = ImageGrab.grab(bbox=(x, y, x1, y1)).convert("L")
        reference_signatures.append(img)                                           

        try:
            processed_img = preprocess_signature(img)
            emb = get_embedding(model, processed_img, device)
        except Exception as e:
            show_notification("Ошибка", f"Ошибка выполнения: {e}", win_register)
            
        reference_signatures_bytes.append(emb.tobytes())
        time_for_signatures.append(np.round((end_time - start_time), decimals=4))   

        canvas.delete("all")
        if(len(reference_signatures) == 5):
            canvas.unbind("<Button-1>")
            canvas.unbind("<ButtonRelease-1>")
            canvas.unbind("<B1-Motion>")
            canvas_container.pack_forget()
            textbox_fio_text = textbox_fio.get() 
            textbox_fio.pack_forget()
            label_enter_fio.configure(text="")
            label_enter_sign.configure(text="")

            camera = cv2.VideoCapture(0)  # ВКЛЮЧАЕМ КАМЕРУ
            if not camera.isOpened():
                raise Exception("Не удалось открыть камеру")
            
            reference_faces = []

            def take_photo():
                success, frame = camera.read()
                if not success:
                    raise Exception("Не удалось получить кадр")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                reference_faces.append(frame)                               
                
                if(len(reference_faces) == 5):
                    btn_take_photo.pack_forget()

                    camera.release()    # ВЫКЛЮЧАЕМ КАМЕРУ

                    label_check = ctk.CTkLabel(register_frame, text="Проверьте данные", font=("Roboto", 24, "bold"), text_color="#2CC985")
                    label_check.pack(pady=(10, 5))

                    label_check_fio = ctk.CTkLabel(register_frame, text=f"ФИО: {textbox_fio_text}", font=("Roboto", 20))
                    label_check_fio.pack(pady=5)

                    label_check_faces = ctk.CTkLabel(register_frame, text="Фото лица:", font=("Roboto", 18, "bold"))
                    label_check_faces.pack(pady=(10, 0))

                    photos_check_frame = ctk.CTkScrollableFrame(register_frame, orientation="horizontal", width=900, height=200, corner_radius=10)
                    photos_check_frame.pack(pady=10)

                    photo_refs = []  # обязательно сохраняем ссылки на изображения
                    for frame_img in reference_faces:
                        ctk_img = ctk.CTkImage(light_image=frame_img, dark_image=frame_img, size=(250, 180))
                        photo_refs.append(ctk_img)
                        lbl_face = ctk.CTkLabel(photos_check_frame, image=ctk_img, text="")
                        lbl_face.pack(side="left", padx=10)

                    label_check_sign = ctk.CTkLabel(register_frame, text="Подпись:", font=("Roboto", 18, "bold"))
                    label_check_sign.pack(pady=(10, 0))

                    sign_check_frame = ctk.CTkScrollableFrame(register_frame, orientation="horizontal", width=900, height=200, corner_radius=10)
                    sign_check_frame.pack(pady=10)

                    sign_refs = []  # обязательно сохраняем ссылки на изображения
                    for sign in reference_signatures:
                        ctk_img = ctk.CTkImage(light_image=sign, dark_image=sign, size=(250, 180))
                        sign_refs.append(ctk_img)
                        lbl_sign = ctk.CTkLabel(sign_check_frame, image=ctk_img, text="")
                        lbl_sign.pack(side="left", padx=10)

                    reference_faces_embeddings = []
                    for face in reference_faces:
                        face_analysis = isf_analysis_faces(face)
                        if(len(face_analysis) == 0):
                            show_notification("Внимание", "Лицо не найдено", win_register)                                                                # ДЛЯ ПЕРСПЕКТИВЫ - НУЖНО ОТКАТЫВАТЬ ПОЛЬЗОВАТЕЛЯ, ЧТОБЫ ОН ПЕРЕДЕЛАЛ ФОТО
                        if(len(face_analysis) > 1):
                            show_notification("Внимание", "В кадре должны быть только ВЫ", win_register)                                                                # ДЛЯ ПЕРСПЕКТИВЫ - НУЖНО ОТКАТЫВАТЬ ПОЛЬЗОВАТЕЛЯ, ЧТОБЫ ОН ПЕРЕДЕЛАЛ ФОТО
                        reference_faces_embeddings.append(face_analysis[0].embedding)            
                       # print(f"\nКол-вол найденных лиц: {len(face_analysis)}")
                       # print(f"\nЭмбеддинги найденных лиц: {face_analysis}")
                       # print("\n\n=================================================================\n")    
                       # print(f"\nСхожесть найденных лиц: {isf_face_batch_similarity([face_emb.embedding for face_emb in face_analysis])}")            

                        # visualize 
                        #isf_faces_visualize(face, face_analysis).show()
                    
                    
                    def send_to_db():
                        try:
                            conn = connect(
                                host="localhost",
                                dbname="Access_db",
                                user="postgres",
                                password="09710972",
                                port="5432"
                            )

                            with conn.cursor() as cursor:
                                cursor.execute("""
                                    INSERT INTO faces (embedding_1, embedding_2, embedding_3, embedding_4, embedding_5)
                                    VALUES (%s, %s, %s, %s, %s)
                                    RETURNING id_face
                                    """, (reference_faces_embeddings[0].tobytes(), 
                                          reference_faces_embeddings[1].tobytes(),
                                          reference_faces_embeddings[2].tobytes(),
                                          reference_faces_embeddings[3].tobytes(),
                                          reference_faces_embeddings[4].tobytes()))
                                id_face = cursor.fetchone()[0]


                                cursor.execute("""
                                    INSERT INTO signatures (embedding_1, embedding_2, embedding_3, embedding_4, embedding_5, time_sign_1, time_sign_2, time_sign_3, time_sign_4, time_sign_5)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                    RETURNING id_signature
                                    """, (reference_signatures_bytes[0],
                                          reference_signatures_bytes[1],
                                          reference_signatures_bytes[2],
                                          reference_signatures_bytes[3],
                                          reference_signatures_bytes[4],
                                          time_for_signatures[0],
                                          time_for_signatures[1],
                                          time_for_signatures[2],
                                          time_for_signatures[3],
                                          time_for_signatures[4]))
                                id_signature = cursor.fetchone()[0]
    
                                data_personal_info = {
                                    'fio':          str(textbox_fio_text),
                                    'id_face':      id_face,
                                    'id_signature': id_signature
                                }
                                cursor.execute("""INSERT INTO personal_info (fio, id_face, id_signature)
                                                  VALUES (%(fio)s, %(id_face)s, %(id_signature)s)
                                                  """, data_personal_info)
                                conn.commit()
                        except Exception as e:
                            if conn:
                                conn.rollback()  # отменяет все изменения в базе данных, которые были сделаны в текущей транзакции, но еще не были подтверждены через conn.commit()
                            show_notification("Ошибка БД", f"Err: {e}", win_register)
                        finally:
                            if conn:
                                conn.close()
                        win_register.destroy()
                        
                        # Окно успешной регистрации
                        win_success_reg = ctk.CTk()
                        win_success_reg.title("Успешно")
                        win_success_reg.geometry("450x200")
                        label_success_reg = ctk.CTkLabel(win_success_reg, text="Вы успешно зарегистрировались!", font=("Roboto", 20, "bold"), text_color="#2CC985")
                        label_success_reg.pack(expand=True)
                        win_success_reg.mainloop()
                                
                    
                    btn_send_to_db = ctk.CTkButton(register_frame, width=220, height=50, text='Все верно', font=("Roboto", 16, "bold"), corner_radius=10, command=send_to_db)      
                    btn_send_to_db.pack(pady=20)

            btn_take_photo = ctk.CTkButton(register_frame, width=220, height=50, text='Сфотографировать', font=("Roboto", 16, "bold"), corner_radius=10, command=take_photo)
            btn_take_photo.pack(pady=20)
            

    def draw(event):
        nonlocal last_x, last_y
        if last_x is None or last_y is None:
            last_x = event.x
            last_y = event.y
            return
        canvas.create_line(last_x, last_y, event.x, event.y, width=5, fill="black", capstyle=tk.ROUND)
        last_x = event.x
        last_y = event.y

    canvas.bind("<Button-1>", start_draw)
    canvas.bind("<ButtonRelease-1>", stop_draw)
    canvas.bind("<B1-Motion>", draw)


# Главное окно программы
win_main = ctk.CTk()
win_main.title("Система доступа")
win_main.geometry('450x350')
win_main.resizable(False, False)

# Контейнер для центрирования элементов главного окна
main_frame = ctk.CTkFrame(win_main, fg_color="transparent")
main_frame.pack(expand=True)

main_lbl = ctk.CTkLabel(main_frame, text="Добро пожаловать", font=("Roboto", 28, "bold"))
main_lbl.pack(pady=(0, 40))

login_btn = ctk.CTkButton(main_frame, width=220, height=50, text='Войти', font=("Roboto", 16, "bold"), corner_radius=10, command=open_win_login)
login_btn.pack(pady=10)

register_btn = ctk.CTkButton(main_frame, width=220, height=50, text='Регистрация', font=("Roboto", 16, "bold"), fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), corner_radius=10, command=open_win_register)
register_btn.pack(pady=10)

win_main.mainloop()