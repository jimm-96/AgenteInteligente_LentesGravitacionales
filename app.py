import os
import numpy as np
import customtkinter as ctkinter
from PIL import Image
import tensorflow as tf

# Configuración estética global (Estilo Cyberpunk / Científico)
ctkinter.set_appearance_mode("Dark")
ctkinter.set_default_color_theme("blue")


class RomanFairApp(ctkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title(" Nancy Grace Roman Telescope - AI Triage Agent")
        self.geometry("1200x700")

        # --- CARGA DE MODELO Y DATOS ---
        self.MODEL_PATH = "cnn_percepcion_pura.h5"
        self.DATASET_PATH = "roman_fair_demo_dataset.npz"

        print("[-] Cargando cerebro de IA local...")
        self.cnn = tf.keras.models.load_model(self.MODEL_PATH)

        # Cargar el banco de imágenes ciegas de la feria
        dataset = np.load(self.DATASET_PATH)
        self.x_test = dataset["data"]
        self.y_test = dataset["labels"]
        self.current_idx = 0

        # Listas en memoria para interacción del experto
        self.imagenes_urgentes = []

        # --- DISEÑO DE LA INTERFAZ (UI) ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. BANNER SUPERIOR / ZONA DE CONTROL
        self.top_panel = ctkinter.CTkFrame(self, height=80, corner_radius=0)
        self.top_panel.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)

        self.title_label = ctkinter.CTkLabel(
            self.top_panel,
            text="SISTEMA DE TRIAJE AUTÓNOMO - AGENTE INTELIGENTE",
            font=ctkinter.CTkFont(size=20, weight="bold"),
        )
        self.title_label.pack(side="left", padx=20)

        self.btn_stream = ctkinter.CTkButton(
            self.top_panel,
            text="Simular Ingesta Masiva (Drop)",
            fg_color="#1f538d",
            hover_color="#143756",
            command=self.start_triage_stream,
            font=ctkinter.CTkFont(size=14, weight="bold"),
        )
        self.btn_stream.pack(side="right", padx=20)

        # 2. COLUMNAS DE CLASIFICACIÓN (ZONAS DE TRIAJE)
        # Columna 1: Archivado Autónomo
        self.col_archived = ctkinter.CTkScrollableFrame(
            self, label_text="ARCHIVADO AUTÓNOMO (P <= 0.35)"
        )
        self.col_archived.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Columna 2: Revisión Secundaria
        self.col_secondary = ctkinter.CTkScrollableFrame(
            self, label_text="REVISIÓN SECUNDARIA (Ambigüedad)"
        )
        self.col_secondary.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        # Columna 3: Despacho Urgente
        self.col_urgent = ctkinter.CTkScrollableFrame(
            self, label_text="DESPACHO URGENTE (P >= 0.85)"
        )
        self.col_urgent.grid(row=1, column=2, sticky="nsew", padx=10, pady=10)

    def start_triage_stream(self):
        """Activa el bucle animado de procesamiento de imágenes"""
        self.btn_stream.configure(state="disabled", text="Procesando Stream...")
        self.process_next_image()

    def process_next_image(self):
        """Procesa una imagen y la envía visualmente a su columna correspondiente"""
        if self.current_idx >= len(self.x_test):
            self.btn_stream.configure(state="normal", text="Ingesta Finalizada")
            return

        # Extraer la matriz numérica actual
        img_matrix = self.x_test[self.current_idx]
        real_label = self.y_test[self.current_idx]

        # Preprocesar al vuelo para la CNN
        tensor = img_matrix.astype("float32") / 255.0
        tensor = np.expand_dims(tensor, axis=0)   # (128, 128) -> (1, 128, 128)
        tensor = np.expand_dims(tensor, axis=-1)  # (1, 128, 128) -> (1, 128, 128, 1)

        # Predicción de la red
        probabilidad = float(self.cnn.predict(tensor, verbose=0)[0][0])

        # Convertir matriz numpy a imagen de CustomTkinter para mostrarla en pantalla
        pil_img = Image.fromarray(img_matrix.astype(np.uint8)).resize((64, 64))
        ctk_img = ctkinter.CTkImage(
            light_image=pil_img, dark_image=pil_img, size=(64, 64)
        )

        # Decisión del Agente: elegir el contenedor destino primero
        if probabilidad <= 0.35:
            parent_col = self.col_archived
        elif probabilidad >= 0.85:
            parent_col = self.col_urgent
        else:
            parent_col = self.col_secondary

        # Crear el botón/tarjeta visual directamente dentro del contenedor correcto
        img_btn = ctkinter.CTkButton(
            parent_col,
            text="",
            image=ctk_img,
            width=64,
            height=64,
            fg_color="transparent",
            hover_color="#2b2b2b",
        )

        # Configurar comportamiento según columna
        if probabilidad <= 0.35:
            img_btn.configure(state="disabled")  # En el archivo no interactúan
        elif probabilidad >= 0.85:
            img_btn.configure(
                command=lambda m=img_matrix, r=real_label: self.open_expert_panel(m, r)
            )
        else:
            img_btn.configure(state="disabled")

        img_btn.pack(pady=5, anchor="center")

        self.current_idx += 1

        # Ajusta este número (en milisegundos) para acelerar o ralentizar la "lluvia" de imágenes
        self.after(50, self.process_next_image)

    def open_expert_panel(self, img_matrix, real_label):
        """Abre una ventana modal interactiva para que el público actúe como Astrónomo Experto"""
        modal = ctkinter.CTkToplevel(self)
        modal.title("Módulo de Validación del Experto Humano")
        modal.geometry("400x500")
        modal.transient(self)  # Forzar foco sobre la app principal
        modal.grab_set()

        # Renderizar la galaxia seleccionada en grande
        pil_large = Image.fromarray(img_matrix.astype(np.uint8)).resize((250, 250))
        ctk_large = ctkinter.CTkImage(
            light_image=pil_large, dark_image=pil_large, size=(250, 250)
        )

        lbl_img = ctkinter.CTkLabel(modal, text="", image=ctk_large)
        lbl_img.pack(pady=20)

        lbl_instruccion = ctkinter.CTkLabel(
            modal,
            text="Analiza la geometría periférica:\n¿Detectas un arco o Anillo de Einstein?",
            font=ctkinter.CTkFont(size=13),
        )
        lbl_instruccion.pack(pady=10)

        # Zona de feedback interactivo
        lbl_feedback = ctkinter.CTkLabel(
            modal, text="", font=ctkinter.CTkFont(size=14, weight="bold")
        )

        def verificar_decision(eleccion_usuario):
            # Criterio científico: real_label == 1 significa que verdaderamente es una lente
            if eleccion_usuario == real_label:
                lbl_feedback.configure(
                    text="¡CORRECTO! Descubrimiento Confirmado 🔭", text_color="green"
                )
            else:
                lbl_feedback.configure(
                    text="ERROR: Clasificación Incorrecta ❌", text_color="red"
                )

            # Deshabilitar botones tras elegir
            btn_si.configure(state="disabled")
            btn_no.configure(state="disabled")

        # Botones de Clasificación del Humano
        btn_si = ctkinter.CTkButton(
            modal,
            text="Confirmar Lente Gravitacional",
            fg_color="#2ecc71",
            hover_color="#27ae60",
            text_color="black",
            command=lambda: verificar_decision(1),
        )
        btn_si.pack(pady=5, fill="x", padx=40)

        btn_no = ctkinter.CTkButton(
            modal,
            text="Marcar como Falso Positivo",
            fg_color="#e74c3c",
            hover_color="#c0392b",
            command=lambda: verificar_decision(0),
        )
        btn_no.pack(pady=5, fill="x", padx=40)

        lbl_feedback.pack(pady=15)

        btn_cerrar = ctkinter.CTkButton(
            modal, text="Cerrar", command=modal.destroy, fg_color="gray"
        )
        btn_cerrar.pack(pady=10)


if __name__ == "__main__":
    app = RomanFairApp()
    app.mainloop()
