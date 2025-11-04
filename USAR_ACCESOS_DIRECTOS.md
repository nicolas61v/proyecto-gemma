# Usar Archivos .BAT como Accesos Directos

## 📊 Análisis de Archivos

### ✅ `instalar_windows_gemma3.bat`
**Estado**: EXCELENTE
- Crea entorno virtual automáticamente
- Instala todas las dependencias
- Verifica errores en cada paso
- **Usar una sola vez** para instalación inicial

### ✅ `ejecutar_gemma3.bat`
**Estado**: MEJORADO
- Ahora verifica que el entorno existe
- Activa automáticamente el `venv`
- Lanza la aplicación
- **Usar cada vez que quieras ejecutar el chatbot**

---

## 🚀 Cómo Crear Accesos Directos en Windows

### Opción 1: Acceso Directo en Escritorio

#### Para Instalar (una sola vez):

1. **Haz clic derecho en `instalar_windows_gemma3.bat`**
   ```
   Enviar a → Escritorio (crear acceso directo)
   ```

2. **Haz clic derecho en el acceso directo → Propiedades**
   ```
   Nombre: ⭐ Instalar Gemma 3 (o el que prefieras)
   Aceptar
   ```

3. **Doble clic para ejecutar**
   - Esperará a terminar
   - Presiona una tecla al finalizar

#### Para Ejecutar (cada vez):

1. **Haz clic derecho en `ejecutar_gemma3.bat`**
   ```
   Enviar a → Escritorio (crear acceso directo)
   ```

2. **Haz clic derecho en el acceso directo → Propiedades**
   ```
   Nombre: 🚀 Abrir Gemma Chat (o el que prefieras
   Aceptar
   ```

3. **Doble clic para ejecutar**
   - Se abre automáticamente el navegador
   - Url: http://127.0.0.1:7860

---

### Opción 2: Carpeta de Acceso Rápido

Windows 10/11 tiene una carpeta especial de acceso rápido:
```
C:\Users\{tu usuario}\AppData\Roaming\Microsoft\Windows\SendTo\
```

Puedes copiar los .bat allí para acceso rápido.

---

## 🎯 Flujo de Uso Recomendado

### Primera Vez (Instalación):

```
1. Doble clic en "Instalar Gemma 3.lnk"
   └─ Espera 5-10 minutos
   └─ Presiona tecla cuando diga "INSTALACION COMPLETADA"

2. Doble clic en "Abrir Gemma Chat.lnk"
   └─ Se abre terminal + navegador automáticamente
```

### Siguientes Veces (Uso):

```
1. Doble clic en "Abrir Gemma Chat.lnk"
   └─ ¡Listo!
```

---

## 📋 Requisitos Previos

Antes de usar los accesos directos:

✅ Python 3.8+ instalado
✅ Git instalado (opcional, pero recomendado)
✅ Conexión a internet (primera ejecución descarga modelo)

---

## ⚠️ Posibles Problemas

### Problema: "No se encuentra el archivo"
**Solución**: Los .bat usan rutas relativas. Mueve el acceso directo a:
```
Escritorio/  (recomendado)
o en la misma carpeta que los .bat
```

### Problema: "No se pudo activar el entorno virtual"
**Solución**: Ejecuta primero `instalar_windows_gemma3.bat` completo

### Problema: "Python no está instalado"
**Solución**:
```
1. Descarga Python 3.8+ desde https://www.python.org/downloads/
2. Instálalo Y marca "Add Python to PATH"
3. Reinicia los .bat
```

### Problema: Tarda mucho en iniciar
**Normal**: Primera ejecución descarga el modelo (~241 MB)
- Segunda ejecución es más rápida

---

## 🎨 Personalizar Accesos Directos

### Cambiar Icono:

1. **Haz clic derecho en el acceso directo → Propiedades**
2. **Botón "Cambiar icono"**
3. Elige un icono bonito (hay muchos en `C:\Windows\System32\`)

Sugerencias:
- Para instalar: 📦 (package icon)
- Para ejecutar: ▶️ (play button)

### Cambiar Nombre:

1. **Haz clic derecho → Cambiar nombre**
2. Escribe un nombre amigable:
   - "⭐ Instalar Gemma 3"
   - "🚀 Abrir Chatbot IA"

---

## ✅ Checklist Final

- [ ] Python 3.8+ instalado
- [ ] `instalar_windows_gemma3.bat` ejecutado sin errores
- [ ] Acceso directo de instalación en Escritorio
- [ ] Acceso directo de ejecución en Escritorio
- [ ] Prueba: Doble clic en "Abrir Gemma Chat"
- [ ] El navegador abre en http://127.0.0.1:7860

---

## 📚 Próximos Pasos

Una vez creados los accesos directos:

1. **Usa "Abrir Gemma Chat" para ejecutar diariamente**
2. **El chatbot estará listo en 10-30 segundos**
3. **Haz preguntas sobre el PDF**
4. **Cierra con Ctrl+C en la terminal**

---

## 🆘 Ayuda

Si tienes problemas:

```bash
1. Abre terminal (Win + R, cmd)
2. Navega a la carpeta del proyecto
3. Ejecuta manualmente:

   instalar_windows_gemma3.bat    (si es la primera vez)
   ejecutar_gemma3.bat             (para ejecutar)

4. Lee los mensajes de error
```

---

## ✨ Resumen

| Archivo | Frecuencia | Propósito |
|---------|-----------|----------|
| instalar_windows_gemma3.bat | 1 sola vez | Setup inicial |
| ejecutar_gemma3.bat | Cada vez | Abrir chatbot |

**Recomendación**: Crea accesos directos para ambos en el Escritorio. Será tu forma rápida de usar el proyecto.

¡Listo! Ahora puedes usar los .bat cómodamente. 🚀
