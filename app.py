import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = (
    "-1"  # Silencia advertencias de GPU no disponible en Windows nativo
)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np

# pyrefly: ignore [missing-import]
import customtkinter as ctkinter
from PIL import Image
import tensorflow as tf

ctkinter.set_appearance_mode("Light")
ctkinter.set_default_color_theme("blue")


class RomanFairAppAdvanced(ctkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title(" Nancy Grace Roman Telescope - AI Triage Agent")
        self.geometry("1280x750")
        self.minsize(900, 600)

        self.running = True
        self.is_streaming = False
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- CONTROL DE DATOS Y MÉTRICAS ---
        self.MODEL_PATH = "cnn_percepcion_pura.h5"
        self.DATASET_PATH = "roman_fair_demo_dataset.npz"

        # Cargar modelo sin compilar (evita advertencia de métricas ya que solo hacemos inferencia)
        self.cnn = tf.keras.models.load_model(self.MODEL_PATH, compile=False)
        dataset = np.load(self.DATASET_PATH)
        self.x_test = dataset["data"]
        self.y_test = dataset["labels"]
        self.current_idx = 0

        # Contadores para el Dashboard en tiempo real
        self.cnt_archived = 0
        self.cnt_secondary = 0
        self.cnt_urgent = 0
        self.total_processed = 0

        # --- CONFIGURACIÓN DE LA GRID GENERAL ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=0)  # Banner
        self.grid_rowconfigure(1, weight=0)  # Dashboard
        self.grid_rowconfigure(2, weight=0)  # Cabeceras personalizadas de columnas
        self.grid_rowconfigure(3, weight=1)  # Columnas scrolleables
        self.grid_rowconfigure(4, weight=0)  # Terminal de logs

        # ============================================================
        # PANEL 1: BANNER SUPERIOR CIENTÍFICO
        # ============================================================
        self.top_panel = ctkinter.CTkFrame(
            self, height=80, corner_radius=8, fg_color=("#e0e7ff", "#1a1a2e")
        )
        self.top_panel.grid(
            row=0, column=0, columnspan=3, sticky="nsew", padx=12, pady=(12, 4)
        )
        self.top_panel.grid_columnconfigure(0, weight=1)
        self.top_panel.grid_columnconfigure(1, weight=0)
        self.top_panel.grid_columnconfigure(2, weight=0)

        self.title_label = ctkinter.CTkLabel(
            self.top_panel,
            text="PROJECT: LENS SENTINEL AI",
            font=ctkinter.CTkFont(family="Arial", size=22, weight="bold"),
            text_color=("#1e3a8a", "#e0e0ff"),
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=(12, 2), sticky="w")

        self.sub_title_label = ctkinter.CTkLabel(
            self.top_panel,
            text="Filtro inteligente y triaje de anomalías cósmicas para el Telescopio Nancy Grace Roman",
            font=ctkinter.CTkFont(family="Arial", size=12, weight="normal"),
            text_color=("#2563eb", "#b0b0d0"),
        )
        self.sub_title_label.grid(row=1, column=0, padx=20, pady=(0, 12), sticky="w")

        self.btn_reset = ctkinter.CTkButton(
            self.top_panel,
            text="🔄 REINICIAR SISTEMA",
            fg_color=("#ef4444", "#5c2424"),
            hover_color=("#dc2626", "#a12d2d"),
            text_color=("white", "white"),
            command=self.reset_system,
            font=ctkinter.CTkFont(size=13, weight="bold"),
            height=38,
            width=180,
            corner_radius=6,
        )
        self.btn_reset.grid(
            row=0, column=1, rowspan=2, padx=(20, 5), pady=10, sticky="e"
        )

        self.btn_stream = ctkinter.CTkButton(
            self.top_panel,
            text="INICIAR INGESTA MASIVA",
            fg_color=("#2563eb", "#1f538d"),
            hover_color=("#1d4ed8", "#2979ff"),
            text_color=("white", "white"),
            command=self.start_triage_stream,
            font=ctkinter.CTkFont(size=13, weight="bold"),
            height=38,
            width=200,
            corner_radius=6,
        )
        self.btn_stream.grid(
            row=0, column=2, rowspan=2, padx=(5, 20), pady=10, sticky="e"
        )

        # ============================================================
        # PANEL 2: DASHBOARD DE TELEMETRÍA (Tarjetas de Métricas)
        # ============================================================
        self.dash_panel = ctkinter.CTkFrame(self, height=55, fg_color="transparent")
        self.dash_panel.grid(
            row=1, column=0, columnspan=3, sticky="nsew", padx=12, pady=4
        )
        self.dash_panel.grid_columnconfigure((0, 1, 2), weight=1)

        # Tarjeta 1: Total Procesado
        self.card_total = ctkinter.CTkLabel(
            self.dash_panel,
            text="📊  Total Procesado: 0",
            font=ctkinter.CTkFont(size=14, weight="bold"),
            fg_color=("#e2e8f0", "#2b2b2b"),
            text_color=("#1e293b", "#e0e0e0"),
            corner_radius=6,
            height=42,
        )
        self.card_total.grid(row=0, column=0, padx=5, sticky="ew")

        # Tarjeta 2: Carga Humana Reducida
        self.card_saved = ctkinter.CTkLabel(
            self.dash_panel,
            text="⚡  Carga Humana Reducida: 0%",
            font=ctkinter.CTkFont(size=14, weight="bold"),
            fg_color=("#dbeafe", "#1e3a5f"),
            text_color=("#2563eb", "#64b5f6"),
            corner_radius=6,
            height=42,
        )
        self.card_saved.grid(row=0, column=1, padx=5, sticky="ew")

        # Tarjeta 3: Alertas a Validar
        self.card_urgent = ctkinter.CTkLabel(
            self.dash_panel,
            text="🚨  Alertas a Validar: 0",
            font=ctkinter.CTkFont(size=14, weight="bold"),
            fg_color=("#dcfce7", "#1b5e20"),
            text_color=("#16a34a", "#a5d6a7"),
            corner_radius=6,
            height=42,
        )
        self.card_urgent.grid(row=0, column=2, padx=5, sticky="ew")

        # ============================================================
        # PANEL 3: COLUMNAS CON CÓDIGO DE COLOR (Cajas del Semáforo)
        # ============================================================
        # --- CABECERA COLUMNA 1 ---
        self.head_archived = ctkinter.CTkFrame(
            self, fg_color=("#e2e8f0", "#1c1c1c"), corner_radius=8
        )
        self.head_archived.grid(row=2, column=0, sticky="ew", padx=(12, 4), pady=(8, 2))
        self.head_archived.grid_columnconfigure(0, weight=1)

        ctkinter.CTkLabel(
            self.head_archived,
            text="⚪ Archivado Autónomo (P ≤ 0.35)",
            font=ctkinter.CTkFont(size=14, weight="bold"),
            text_color=("#475569", "#9e9e9e"),
        ).grid(row=0, column=0, padx=12, pady=(8, 2), sticky="w")

        ctkinter.CTkLabel(
            self.head_archived,
            text="Galaxias normales • IA las descarta sola • ~50% Ahorro",
            font=ctkinter.CTkFont(size=10, weight="normal"),
            text_color=("#64748b", "#757575"),
        ).grid(row=1, column=0, padx=12, pady=(0, 8), sticky="w")

        # --- COLUMNA 1: Archivado (Gris oscuro) ---
        self.col_archived = ctkinter.CTkScrollableFrame(
            self,
            label_text="",
            fg_color=("#f8fafc", "#1c1c1c"),
            corner_radius=8,
        )
        self.col_archived.grid(
            row=3, column=0, sticky="nsew", padx=(12, 4), pady=(2, 12)
        )

        # --- CABECERA COLUMNA 2 ---
        self.head_secondary = ctkinter.CTkFrame(
            self, fg_color=("#fef3c7", "#1e1a0e"), corner_radius=8
        )
        self.head_secondary.grid(row=2, column=1, sticky="ew", padx=4, pady=(8, 2))
        self.head_secondary.grid_columnconfigure(0, weight=1)

        ctkinter.CTkLabel(
            self.head_secondary,
            text="🟡 Revisión Secundaria (Ambigüedad)",
            font=ctkinter.CTkFont(size=14, weight="bold"),
            text_color=("#d97706", "#ffb300"),
        ).grid(row=0, column=0, padx=12, pady=(8, 2), sticky="w")

        ctkinter.CTkLabel(
            self.head_secondary,
            text="Casos inciertos o con ruido • Aislados por precaución",
            font=ctkinter.CTkFont(size=10, weight="normal"),
            text_color=("#b45309", "#d69e2e"),
        ).grid(row=1, column=0, padx=12, pady=(0, 8), sticky="w")

        # --- COLUMNA 2: Revisión Secundaria (Ámbar) ---
        self.col_secondary = ctkinter.CTkScrollableFrame(
            self,
            label_text="",
            fg_color=("#fffbeb", "#1e1a0e"),
            corner_radius=8,
        )
        self.col_secondary.grid(row=3, column=1, sticky="nsew", padx=4, pady=(2, 12))

        # --- CABECERA COLUMNA 3 ---
        self.head_urgent = ctkinter.CTkFrame(
            self, fg_color=("#dcfce7", "#0d1f14"), corner_radius=8
        )
        self.head_urgent.grid(row=2, column=2, sticky="ew", padx=(4, 12), pady=(8, 2))
        self.head_urgent.grid_columnconfigure(0, weight=1)

        ctkinter.CTkLabel(
            self.head_urgent,
            text="🟢 Despacho Crítico (P ≥ 0.85)",
            font=ctkinter.CTkFont(size=14, weight="bold"),
            text_color=("#16a34a", "#2ecc71"),
        ).grid(row=0, column=0, padx=12, pady=(8, 2), sticky="w")

        ctkinter.CTkLabel(
            self.head_urgent,
            text="Candidatos a lentes • ¡Haga clic para validar como experto!",
            font=ctkinter.CTkFont(size=10, weight="bold"),
            text_color=("#15803d", "#27ae60"),
        ).grid(row=1, column=0, padx=12, pady=(0, 8), sticky="w")

        # --- COLUMNA 3: Despacho Crítico (Verde) ---
        self.col_urgent = ctkinter.CTkScrollableFrame(
            self,
            label_text="",
            fg_color=("#f0fdf4", "#0d1f14"),
            corner_radius=8,
        )
        self.col_urgent.grid(row=3, column=2, sticky="nsew", padx=(4, 12), pady=(2, 12))

        # ============================================================
        # PANEL 4: TERMINAL DE ESTADO / LOGS
        # ============================================================
        self.log_panel = ctkinter.CTkFrame(
            self,
            height=35,
            corner_radius=8,
            fg_color=("#f1f5f9", "#111118"),
            border_width=1,
            border_color=("#cbd5e1", "#1f1f3a"),
        )
        self.log_panel.grid(
            row=4, column=0, columnspan=3, sticky="ew", padx=12, pady=(4, 12)
        )
        self.log_panel.grid_columnconfigure(0, weight=1)

        self.lbl_status_log = ctkinter.CTkLabel(
            self.log_panel,
            text="[SISTEMA] Listo para iniciar ingesta y triaje de datos.",
            font=ctkinter.CTkFont(family="Courier New", size=12, weight="bold"),
            text_color=("#166534", "#88a888"),
        )
        self.lbl_status_log.grid(row=0, column=0, padx=15, pady=5, sticky="w")

    def start_triage_stream(self):
        self.is_streaming = True
        self.btn_stream.configure(state="disabled", text="ANALIZANDO FLUJO...")
        self.process_next_image()

    def update_dashboard(self):
        """Actualiza las tarjetas métricas del sistema en tiempo real"""
        self.card_total.configure(text=f"📊  Total Procesado: {self.total_processed}")
        porcentaje_ahorro = (
            (self.cnt_archived / self.total_processed * 100)
            if self.total_processed > 0
            else 0
        )
        # Cambiar color de la tarjeta de ahorro dinámicamente
        if porcentaje_ahorro >= 50:
            self.card_saved.configure(
                fg_color=("#dbeafe", "#1e3a5f"),
                text_color=("#2563eb", "#64b5f6"),
                text=f"⚡  Carga Humana Reducida: {porcentaje_ahorro:.1f}%",
            )
        else:
            self.card_saved.configure(
                fg_color=("#e2e8f0", "#2b2b2b"),
                text_color=("#475569", "#b0bec5"),
                text=f"⚡  Carga Humana Reducida: {porcentaje_ahorro:.1f}%",
            )
        self.card_urgent.configure(text=f"🚨  Alertas a Validar: {self.cnt_urgent}")

    def process_next_image(self):
        if not self.running or not self.is_streaming:
            return

        if self.current_idx >= len(self.x_test):
            self.is_streaming = False
            self.btn_stream.configure(
                text="✓ FLUJO COMPLETADO",
                fg_color=("#16a34a", "#1b5e20"),
                state="disabled",
            )
            return

        img_matrix = self.x_test[self.current_idx]
        real_label = self.y_test[self.current_idx]

        tensor = img_matrix.astype("float32") / 255.0
        tensor = np.expand_dims(tensor, axis=(0, -1))
        probabilidad = float(self.cnn.predict(tensor, verbose=0)[0][0])

        pil_img = Image.fromarray(img_matrix).resize((70, 70))
        ctk_img = ctkinter.CTkImage(
            light_image=pil_img, dark_image=pil_img, size=(70, 70)
        )

        self.total_processed += 1

        # Asignación y lógica visual basada en colores
        if probabilidad <= 0.35:
            self.cnt_archived += 1
            self.log_status(
                f"[SISTEMA] Muestra #{self.total_processed} clasificada como ARCHIVADO (P = {probabilidad:.3f})"
            )
            img_btn = ctkinter.CTkButton(
                self.col_archived,
                text="",
                image=ctk_img,
                width=70,
                height=70,
                fg_color=("#e2e8f0", "#2b2b2b"),
                hover_color=("#cbd5e1", "#3a3a3a"),
                border_width=2,
                border_color=("#94a3b8", "#555555"),
                corner_radius=6,
                cursor="hand2",
            )
            img_btn.configure(
                command=lambda btn=img_btn, m=img_matrix, r=real_label, p=probabilidad, col="archived": self.open_expert_panel(
                    btn, m, r, p, col
                )
            )
            img_btn.image = (
                ctk_img  # Mantener referencia para evitar garbage collection
            )
            img_btn.pack(pady=6, anchor="center")
        elif probabilidad >= 0.85:
            self.cnt_urgent += 1
            self.log_status(
                f"[SISTEMA] Muestra #{self.total_processed} clasificada como DESPACHO CRÍTICO (P = {probabilidad:.3f})"
            )
            img_btn = ctkinter.CTkButton(
                self.col_urgent,
                text="",
                image=ctk_img,
                width=70,
                height=70,
                fg_color=("#dcfce7", "#1e3d24"),
                hover_color=("#bbf7d0", "#2e6b38"),
                border_width=2,
                border_color=("#22c55e", "#2ecc71"),
                corner_radius=6,
                cursor="hand2",
            )
            img_btn.configure(
                command=lambda btn=img_btn, m=img_matrix, r=real_label, p=probabilidad, col="urgent": self.open_expert_panel(
                    btn, m, r, p, col
                )
            )
            img_btn.image = (
                ctk_img  # Mantener referencia para evitar garbage collection
            )
            img_btn.pack(pady=6, anchor="center")
        else:
            self.cnt_secondary += 1
            self.log_status(
                f"[SISTEMA] Muestra #{self.total_processed} clasificada como REVISIÓN SECUNDARIA (P = {probabilidad:.3f})"
            )
            img_btn = ctkinter.CTkButton(
                self.col_secondary,
                text="",
                image=ctk_img,
                width=70,
                height=70,
                fg_color=("#fef3c7", "#2a2010"),
                hover_color=("#fde68a", "#3a3a3a"),
                border_width=2,
                border_color=("#f59e0b", "#ffb300"),
                corner_radius=6,
                cursor="hand2",
            )
            img_btn.configure(
                command=lambda btn=img_btn, m=img_matrix, r=real_label, p=probabilidad, col="secondary": self.open_expert_panel(
                    btn, m, r, p, col
                )
            )
            img_btn.image = (
                ctk_img  # Mantener referencia para evitar garbage collection
            )
            img_btn.pack(pady=6, anchor="center")

        self.update_dashboard()
        self.current_idx += 1
        self.after(40, self.process_next_image)

    def log_status(self, message):
        self.lbl_status_log.configure(text=message)

    def open_expert_panel(self, btn, img_matrix, real_label, probabilidad, column_type):
        modal = ctkinter.CTkToplevel(self)
        modal.title("Inspección de Muestra")
        modal.geometry("420x640")
        modal.transient(self)
        modal.grab_set()
        modal.resizable(False, False)

        # Configurar colores del modal según la procedencia
        if column_type == "archived":
            modal_title = "🔍  Inspección de Archivado"
            header_fg = ("#e2e8f0", "#222222")
            text_color = ("#475569", "#9e9e9e")
            instruccion = "¿Confirma que es una GALAXIA NORMAL?"
            btn_si_text = "✔  CONFIRMAR (GALAXIA NORMAL)"
            btn_no_text = "⚠  RECLASIFICAR COMO LENTE"
        elif column_type == "secondary":
            modal_title = "🟡  Inspección de Revisión"
            header_fg = ("#fef3c7", "#1e1a0e")
            text_color = ("#d97706", "#ffb300")
            instruccion = "¿Confirma curvatura de lente gravitacional?"
            btn_si_text = "✔  CONFIRMAR LENTE"
            btn_no_text = "✖  RECHAZAR (GALAXIA NORMAL)"
        else:  # urgent
            modal_title = "🔭  Validación de Alerta"
            header_fg = ("#dcfce7", "#0d1f14")
            text_color = ("#15803d", "#2ecc71")
            instruccion = "¿Confirma curvatura de lente gravitacional?"
            btn_si_text = "✔  CONFIRMAR LENTE"
            btn_no_text = "✖  RECHAZAR (FALSO POSITIVO)"

        # Encabezado del modal
        header = ctkinter.CTkFrame(
            modal, fg_color=header_fg, corner_radius=0, height=50
        )
        header.pack(fill="x")
        ctkinter.CTkLabel(
            header,
            text=modal_title,
            font=ctkinter.CTkFont(size=16, weight="bold"),
            text_color=text_color,
        ).pack(pady=12)

        pil_large = Image.fromarray(img_matrix).resize((260, 250))
        ctk_large = ctkinter.CTkImage(
            light_image=pil_large, dark_image=pil_large, size=(260, 250)
        )

        lbl_img = ctkinter.CTkLabel(modal, text="", image=ctk_large)
        lbl_img.image = ctk_large  # Mantener referencia para evitar garbage collection
        lbl_img.pack(pady=12)

        lbl_prob = ctkinter.CTkLabel(
            modal,
            text=f"Probabilidad de Lente (IA): P = {probabilidad:.4f}",
            font=ctkinter.CTkFont(size=12, weight="bold"),
            text_color=("#475569", "#a0b0d0"),
        )
        lbl_prob.pack(pady=(0, 5))

        lbl_instruccion = ctkinter.CTkLabel(
            modal,
            text=instruccion,
            font=ctkinter.CTkFont(size=13, weight="bold"),
            text_color=("#0f172a", "#e0e0e0"),
        )
        lbl_instruccion.pack(pady=(0, 10))

        lbl_feedback = ctkinter.CTkLabel(
            modal, text="", font=ctkinter.CTkFont(size=14, weight="bold")
        )

        def verificar_decision(eleccion_usuario):
            if column_type == "archived":
                es_correcto = eleccion_usuario == real_label
            else:
                es_correcto = eleccion_usuario == real_label

            if es_correcto:
                if real_label == 1:
                    if column_type == "urgent":
                        lbl_feedback.configure(
                            text="¡CORRECTO! 🔭\nSe ha confirmado la lente gravitacional.",
                            text_color=("#16a34a", "#2ecc71"),
                        )
                        self.log_status(
                            f"[EXPERTO] Muestra #{self.current_idx} validada: LENTE CONFIRMADO (¡Correcto! Predicción verificada)"
                        )
                        btn.configure(
                            border_color=("#16a34a", "#2ecc71"),
                            fg_color=("#dcfce7", "#1e3d24"),
                            state="disabled",
                        )
                    elif column_type == "archived":
                        lbl_feedback.configure(
                            text="¡DESCUBRIMIENTO RESCATADO! 🔭\nCorregido falso negativo de archivado.",
                            text_color=("#2563eb", "#3498db"),
                        )
                        self.log_status(
                            f"[EXPERTO] Muestra #{self.current_idx} ¡Rescate de lente! Corregido falso negativo de archivado."
                        )
                        btn.configure(
                            border_color=("#2563eb", "#3498db"),
                            fg_color=("#e0f2fe", "#1a2f4c"),
                            state="disabled",
                        )
                    else:  # secondary
                        lbl_feedback.configure(
                            text="¡RESOLUCIÓN CORRECTA! 🔭\nMuestra de Revisión validada como Lente.",
                            text_color=("#2563eb", "#3498db"),
                        )
                        self.log_status(
                            f"[EXPERTO] Muestra #{self.current_idx} Ambigüedad resuelta: LENTE CONFIRMADO en Revisión."
                        )
                        btn.configure(
                            border_color=("#2563eb", "#3498db"),
                            fg_color=("#e0f2fe", "#1a2f4c"),
                            state="disabled",
                        )
                else:  # real_label == 0
                    if column_type == "urgent":
                        lbl_feedback.configure(
                            text="¡CORRECCIÓN EXITOSA! ❌\nFalso positivo de la IA identificado y corregido.",
                            text_color=("#dc2626", "#e74c3c"),
                        )
                        self.log_status(
                            f"[EXPERTO] Muestra #{self.current_idx} Falso positivo de la IA filtrado y corregido."
                        )
                        btn.configure(
                            border_color=("#dc2626", "#e74c3c"),
                            fg_color=("#e2e8f0", "#2b2b2b"),
                            state="disabled",
                        )
                    elif column_type == "archived":
                        lbl_feedback.configure(
                            text="¡CORRECTO! ⚪\nSe ha confirmado que es una galaxia normal.",
                            text_color=("#475569", "#9e9e9e"),
                        )
                        self.log_status(
                            f"[EXPERTO] Muestra #{self.current_idx} confirmada como Galaxia Normal."
                        )
                        btn.configure(
                            border_color=("#94a3b8", "#555555"),
                            fg_color=("#f1f5f9", "#1c1c1c"),
                            state="disabled",
                        )
                    else:  # secondary
                        lbl_feedback.configure(
                            text="¡RESOLUCIÓN CORRECTA! ⚪\nMuestra de Revisión confirmada como Galaxia Normal.",
                            text_color=("#475569", "#9e9e9e"),
                        )
                        self.log_status(
                            f"[EXPERTO] Muestra #{self.current_idx} Ambigüedad resuelta: confirmada Galaxia Normal."
                        )
                        btn.configure(
                            border_color=("#94a3b8", "#555555"),
                            fg_color=("#f1f5f9", "#1c1c1c"),
                            state="disabled",
                        )
            else:
                if real_label == 1:
                    lbl_feedback.configure(
                        text="ALERTA: ERROR DE VALIDACIÓN ❌\nEsta muestra contiene una lente gravitacional real.",
                        text_color=("#dc2626", "#e74c3c"),
                    )
                    self.log_status(
                        f"[ERROR EXPERTO] Muestra #{self.current_idx} descartada erróneamente (Contiene Lente)."
                    )
                else:
                    lbl_feedback.configure(
                        text="ALERTA: ERROR DE VALIDACIÓN ❌\nEsta muestra es en realidad una galaxia normal.",
                        text_color=("#dc2626", "#e74c3c"),
                    )
                    self.log_status(
                        f"[ERROR EXPERTO] Muestra #{self.current_idx} clasificada erróneamente (Es Galaxia Normal)."
                    )
                btn.configure(
                    border_color=("#dc2626", "#e74c3c"),
                    fg_color=("#fee2e2", "#5c2424"),
                    state="disabled",
                )

            # Consecuencia en contadores: decrementar si era alerta urgente pendiente
            if column_type == "urgent":
                self.cnt_urgent = max(0, self.cnt_urgent - 1)
                self.update_dashboard()

            btn_si.configure(state="disabled")
            btn_no.configure(state="disabled")

        btn_si = ctkinter.CTkButton(
            modal,
            text=btn_si_text,
            fg_color="#2ecc71",
            hover_color="#27ae60",
            text_color="black",
            command=lambda: verificar_decision(0 if column_type == "archived" else 1),
            font=ctkinter.CTkFont(size=13, weight="bold"),
            height=40,
            corner_radius=6,
        )
        btn_si.pack(pady=5, fill="x", padx=40)

        btn_no = ctkinter.CTkButton(
            modal,
            text=btn_no_text,
            fg_color="#3498db" if column_type == "archived" else "#e74c3c",
            hover_color="#2980b9" if column_type == "archived" else "#c0392b",
            text_color="white" if column_type == "archived" else "black",
            command=lambda: verificar_decision(1 if column_type == "archived" else 0),
            font=ctkinter.CTkFont(size=13, weight="bold"),
            height=40,
            corner_radius=6,
        )
        btn_no.pack(pady=5, fill="x", padx=40)

        lbl_feedback.pack(pady=10)

        ctkinter.CTkButton(
            modal,
            text="Cerrar",
            command=modal.destroy,
            fg_color=("#64748b", "#424242"),
            hover_color=("#475569", "#616161"),
            text_color=("white", "white"),
            height=34,
            corner_radius=6,
        ).pack(pady=5, padx=40, fill="x")

    def reset_system(self):
        # Detener el flujo si está activo
        self.is_streaming = False

        # Destruir widgets hijos de las columnas
        for widget in self.col_archived.winfo_children():
            widget.destroy()
        for widget in self.col_secondary.winfo_children():
            widget.destroy()
        for widget in self.col_urgent.winfo_children():
            widget.destroy()

        # Reiniciar contadores e índice
        self.cnt_archived = 0
        self.cnt_secondary = 0
        self.cnt_urgent = 0
        self.total_processed = 0
        self.current_idx = 0

        # Actualizar telemetría del dashboard
        self.update_dashboard()
        self.log_status("[SISTEMA] Listo para iniciar ingesta y triaje de datos.")

        # Restablecer el botón de ingesta
        self.btn_stream.configure(
            state="normal",
            text="INICIAR INGESTA MASIVA",
            fg_color=("#2563eb", "#1f538d"),
        )

    def on_closing(self):
        self.running = False
        self.destroy()


if __name__ == "__main__":
    app = RomanFairAppAdvanced()
    app.mainloop()
