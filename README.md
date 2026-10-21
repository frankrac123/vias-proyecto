# Sistema Profesional de Aforo Vehicular y Peatonal

## 📋 Descripción

Sistema profesional de análisis y conteo de tráfico vehicular y peatonal en tiempo real utilizando visión por computadora (YOLO) y seguimiento de objetos. Diseñado para funcionar localmente con cámaras IP o archivos de video.

**Características:**
- Detección en tiempo real con YOLO11
- Seguimiento robusto de objetos con ByteTrack
- Conteo inteligente basado en geometría de líneas
- Análisis de flujo vehicular por intervalos
- Almacenamiento en base de datos SQLite
- Interfaz gráfica moderna con PySide6
- Exportación de datos en Excel y CSV

## 🔧 Requisitos del Sistema

### Hardware
- **CPU**: Intel i5/i7 o equivalente AMD
- **RAM**: Mínimo 8GB (recomendado 16GB)
- **GPU (opcional)**: NVIDIA CUDA-compatible para mejor rendimiento
- **Almacenamiento**: 10GB libres

### Software
- Python 3.9 o superior
- Windows 10/11 o Linux

## 📦 Instalación

### 1. Preparar el Entorno Virtual

```bash
cd c:\Detector_Trafico

# Crear entorno virtual si no existe
python -m venv .venv

# Activar entorno virtual
.venv\Scripts\activate
```

### 2. Instalar Dependencias

```bash
# Instalar desde requirements.txt
pip install -r requirements.txt

# Si hay problemas con torch, instalar manualmente:
# CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verificar Instalación

```bash
python -c "import cv2, ultralytics, PySide6; print('Todas las librerías instaladas correctamente')"
```

## 🚀 Ejecución

### Iniciar la Aplicación

```bash
python main.py
```

La aplicación abrirá una ventana con:
- Panel de video en vivo (izquierda)
- Panel de estadísticas (derecha)
- Controles y configuración

## ⚙️ Configuración

### Configurar Cámara IP (Android con IP Webcam)

1. **Instalar IP Webcam** en el dispositivo Android
2. **Iniciar el servidor** en la app
3. **Obtener la URL**:
   - MJPEG: `http://IP_DE_TU_CAMARA:8080/video`
   - RTSP: `rtsp://IP_DE_TU_CAMARA:8080/h264.sdp`

### Configuración de la Aplicación

En la interfaz:

1. **Botón ⚙ CONFIGURACIÓN**
2. Establecer:
   - **URL Cámara**: la URL de tu propia cámara o un archivo de video
   - **Tipo de Fuente**: `mjpeg` (o `rtsp`, `file`, `usb`)
   - **Confianza**: 0.30 (ajustar según necesidad)
   - **Tamaño de Imagen**: 416 (balanceado) o 640 (calidad) o 320 (rendimiento)
   - **Dispositivo**: `auto` (detecta CUDA automáticamente)

### Archivo de Configuración

Se crea automáticamente en `config/config.json`:

```json
{
  "camera": {
    "source": "0",
    "source_type": "usb",
    "frame_buffer_size": 2,
    "reconnect_attempts": 5,
    "reconnect_delay": 2.0
  },
  "detection": {
    "model_path": "yolo11n.pt",
    "confidence": 0.30,
    "imgsz": 416,
    "device": "auto",
    "classes": [0, 1, 2, 3, 5, 7]
  },
  "line": {
    "x1": 0.5,
    "y1": 0.0,
    "x2": 0.5,
    "y2": 1.0,
    "name": "Línea Principal"
  }
}
```

## 📏 Configurar Línea de Conteo

### Método 1: Editor Visual

1. Botón **📏 EDITAR LÍNEA**
2. Seleccionar preset:
   - **Línea Horizontal**: A través del ancho del video
   - **Línea Vertical**: De arriba a abajo
   - **Línea Diagonal**: En diagonal

### Método 2: Edición Manual

1. Abrir **config/config.json**
2. Modificar sección `line`:
   ```json
   "line": {
     "x1": 0.0,      // Posición X inicio (0.0 - 1.0)
     "y1": 0.5,      // Posición Y inicio
     "x2": 1.0,      // Posición X fin
     "y2": 0.5,      // Posición Y fin
     "name": "Mi Línea Personalizada"
   }
   ```

## 📊 Interpretación de Datos

### Volumen
- **Definición**: Cantidad de objetos que cruzaron la línea de conteo
- **Uso**: Determinar intensidad de tráfico en un período

### Flujo
- **Definición**: Cantidad de objetos por unidad de tiempo
- **Expresión**: veh/min o veh/h
- **Uso**: Evaluar velocidad de tránsito y congestión

### Categorías
- **Persona**: Peatones
- **Carro**: Automóviles
- **Moto**: Motocicletas
- **Bicicleta**: Ciclos
- **Bus**: Autobús o transporte masivo
- **Camión**: Vehículos pesados

### Sentidos
- **A→B**: Objetos cruzando de lado A a lado B
- **B→A**: Objetos cruzando en dirección opuesta

## 📤 Exportación de Datos

### Formatos Soportados

1. **Excel (XLSX)**
   - Hoja "Resumen": Estadísticas generales
   - Hoja "Eventos": Detalle de cada cruce
   - Hoja "Analítica": Datos por intervalo (1, 5, 15, 60 minutos)
   - Hoja "Categorías": Distribución por clase
   - Hoja "Sentidos": Distribución por dirección

2. **CSV**
   - Archivo con todos los eventos
   - Compatible con Excel, Python, R, etc.

### Proceso de Exportación

1. Pestaña **📤 Exportar**
2. Seleccionar formatos (CSV y/o XLSX)
3. Botón **📤 EXPORTAR**
4. Los archivos se guardan en `./exports/`

## 🔍 Solución de Problemas

### La aplicación no abre
```bash
# Verificar dependencias
pip check

# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

### Error: "No se pudo conectar a la cámara"
- Verificar que la URL es correcta
- Probar la URL propia de la cámara en el navegador
- Verificar que el dispositivo está en la misma red
- Si es MJPEG, confirmar que el servidor está activo

### FPS bajo
- Reducir confianza (menos detecciones)
- Aumentar tamaño de imagen (imgsz) en configuración
- Usar modo de rendimiento ("Rendimiento" en config)
- Habilitar GPU si está disponible

### Conteos incorrectos
- Ajustar ángulo de línea (más perpendicular al flujo)
- Aumentar confianza mínima si hay falsos positivos
- Verificar que la línea cruza completamente el flujo

### Latencia alta
- Verificar conexión de red (usar cableado Ethernet)
- Reducir resolución de la cámara
- Usar RTSP en lugar de MJPEG si está disponible
- Aumentar imgsz si la GPU está disponible

## 📊 Rendimiento

### Comparación de Modos

| Modo | imgsz | Velocidad | Precisión | GPU |
|------|-------|-----------|-----------|-----|
| Rendimiento | 320 | ⭐⭐⭐⭐⭐ | ⭐⭐ | Opcional |
| Balanceado | 416 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Recomendado |
| Calidad | 640 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Recomendado |

### Rendimiento Esperado

- **CPU (sin GPU)**: 8-15 FPS
- **GPU NVIDIA (CUDA)**: 25-60 FPS
- **Latencia**: 30-150 ms (según modo)

## 🗄️ Base de Datos

### Ubicación
`./data/traffic.db` (SQLite)

### Tablas

- **sessions**: Información de sesiones
- **crossing_events**: Cada cruce registrado
- **analytics_intervals**: Estadísticas por intervalo

### Acceso

```python
# Python
import sqlite3

conn = sqlite3.connect('./data/traffic.db')
cursor = conn.cursor()

# Ver eventos
cursor.execute('SELECT * FROM crossing_events LIMIT 10')
for row in cursor.fetchall():
    print(row)
```

## 📝 Estructura del Proyecto

```
Detector_Trafico/
├── main.py                 # Punto de entrada
├── requirements.txt        # Dependencias
├── README.md              # Este archivo
├── detector_legacy.py     # Código original (backup)
│
├── app/
│   ├── __init__.py
│   ├── gui.py             # Interfaz gráfica (PySide6)
│   ├── main_processor.py  # Procesamiento principal
│   ├── video_capture.py   # Captura asíncrona de video
│   ├── detector.py        # Wrapper de YOLO
│   ├── tracker.py         # Gestor de seguimiento
│   ├── counting.py        # Lógica de conteo con geometría
│   ├── analytics.py       # Motor de análisis
│   ├── database.py        # Gestor SQLite
│   ├── exporter.py        # Exportación Excel/CSV
│   ├── config.py          # Gestión de configuración
│   └── utils.py           # Funciones auxiliares
│
├── config/
│   └── config.json        # Configuración (auto-generada)
│
├── data/
│   └── traffic.db         # Base de datos SQLite
│
├── exports/               # Archivos exportados
│   ├── session_*.xlsx
│   └── events_*.csv
│
└── sessions/              # Datos de sesiones
```

## 🎯 Casos de Uso

### 1. Monitoreo de Intersección Vial
- Configurar línea vertical u horizontal
- Monitorear volumen por dirección
- Generar reportes horarios

### 2. Análisis de Flujo Peatonal
- Colocar cámara perpendicular al flujo
- Contar peatones por intervalo
- Identificar horas pico

### 3. Gestión de Estacionamiento
- Contar vehículos que entran/salen
- Monitorear ocupación en tiempo real
- Exportar datos para reportes

### 4. Estudios de Tráfico
- Recolectar datos a largo plazo
- Analizar patrones por hora/día
- Exportar para análisis estadístico

## 🔐 Consideraciones de Seguridad

- No se almacenan videos, solo eventos de conteo
- Base de datos local, sin transmisión a servidores
- Configuración almacenada localmente
- Compatible con cámaras en red privada

## 📞 Soporte

### Recursos
- [YOLO Documentation](https://docs.ultralytics.com)
- [PySide6 Documentation](https://doc.qt.io/qtforpython)
- [OpenCV Documentation](https://docs.opencv.org)

### Verificar Instalación

```bash
# Verificar CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Verificar YOLO
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Verificar PySide6
python -c "from PySide6 import QtWidgets; print('OK')"
```

## 📄 Licencia

Sistema profesional de tráfico. Uso interno.

## 🚀 Próximas Mejoras

- [ ] Soporte para múltiples líneas de conteo
- [ ] Análisis de velocidad de vehículos
- [ ] Reconocimiento de placas (OCR)
- [ ] Alertas de congestión automática
- [ ] Dashboard web en tiempo real
- [ ] Integración con bases de datos remotas

---

**Versión**: 1.0.0  
**Última actualización**: 2024-2025  
**Estado**: Producción
