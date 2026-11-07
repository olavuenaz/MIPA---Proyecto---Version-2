# MIPA — Modelo de Inteligencia Artificial con Personalidad Adaptable

**Prototipo de investigación | Inteligencia Artificial | Procesamiento del Lenguaje Natural | Personalidad Computacional**

MIPA es un prototipo experimental de sistema conversacional basado en modelos de lenguaje tipo Transformer que explora la adaptación de respuestas mediante la integración de modelos psicológicos de personalidad, mecanismos de inferencia contextual, lógica difusa y restricciones éticas operacionalizadas.

El proyecto investiga cómo representar características de personalidad dentro de una arquitectura computacional modular, utilizando el modelo de los Cinco Grandes Factores (*Big Five/OCEAN*) y arquetipos inspirados en la psicología analítica de Carl Gustav Jung.

MIPA no pretende reproducir la personalidad humana ni atribuir estados psicológicos reales a un modelo de lenguaje. Su objetivo es estudiar mecanismos computacionales que permitan generar respuestas con estilos diferenciados, coherencia contextual y restricciones explícitas de comportamiento.

## 1. Problema de investigación

La investigación se organiza en torno a la siguiente pregunta:

> ¿Cómo puede diseñarse e implementarse una arquitectura de sistema conversacional basado en Transformers que integre un motor de inferencia psicológica mediante lógica difusa y mecanismos de gobernanza ética, para generar respuestas adaptables al contexto, consistentes según la coherencia arquetípica, y alineadas con restricciones éticas operacionalizadas, evaluadas en un entorno controlado y simulado?

## 2. Objetivo general

Diseñar y evaluar conceptualmente una arquitectura modular de inteligencia artificial con personalidad adaptable, integrando fundamentos psicológicos y computacionales mediante un prototipo experimental de interacción conversacional.

### Objetivos específicos

- Examinar principios éticos aplicables al diseño de sistemas conversacionales con adaptación de personalidad.
- Establecer una representación computacional de rasgos de personalidad basada en Big Five/OCEAN y su relación conceptual con arquetipos junguianos.
- Diseñar una arquitectura modular que permita separar percepción, inferencia, adaptación, generación y evaluación.
- Implementar un prototipo funcional que permita observar el comportamiento de diferentes configuraciones de personalidad.
- Evaluar experimentalmente la coherencia de las respuestas y el comportamiento del sistema bajo escenarios controlados.

## 3. Arquitectura del sistema

MIPA utiliza una arquitectura modular para separar el procesamiento de las entradas, la interpretación contextual, la selección del comportamiento y la generación de respuestas.

```text
                         USUARIO
                            |
                            v
                  Interfaz Streamlit
                       app_ui.py
                            |
                            v
                    API FastAPI
                       main.py
                            |
                            v
                 Percepción contextual
                modules/perception.py
                            |
                            v
                 Inferencia y decisión
                modules/reasoning.py
                            |
                 +----------+----------+
                 |                     |
                 v                     v
          Perfil OCEAN          Reglas y restricciones
          y arquetipos              de seguridad
                 |                     |
                 +----------+----------+
                            |
                            v
                 Generación de respuesta
                modules/generation.py
                            |
                            v
                    Modelos Ollama
                            |
                            v
                    Respuesta final
                            |
                            v
                       USUARIO
```

Este diagrama representa el flujo conversacional principal a nivel conceptual. Otros componentes están implementados en el repositorio, pero no necesariamente participan en todas las solicitudes.

### Componentes principales

| Archivo | Responsabilidad |
|---|---|
| `main.py` | Orquestación de la aplicación y exposición de endpoints mediante FastAPI. |
| `app_ui.py` | Interfaz conversacional y controles de visualización desarrollados con Streamlit. |
| `modules/perception.py` | Procesamiento e interpretación contextual de las entradas. |
| `modules/reasoning.py` | Inferencia y selección de comportamientos a partir del contexto y las configuraciones. |
| `modules/generation.py` | Construcción de solicitudes y generación de respuestas mediante modelos locales. |
| `modules/audit.py` | Registro y trazabilidad de decisiones del sistema. |
| `modules/memory.py` | Funcionalidades de memoria y recuperación de información. |
| `modules/input.py` | Funcionalidades complementarias de procesamiento de entradas. |
| `modules/evaluation.py` | Funcionalidades de evaluación del comportamiento. |

**Estado de integración:** el flujo principal utiliza componentes de percepción, razonamiento y generación. Los módulos de memoria, entrada y evaluación no deben interpretarse como subsistemas completamente integrados en todas las rutas de ejecución. En particular, la recuperación de memoria no está conectada como contexto recuperado en el flujo principal de `/chat` de esta versión.

## 4. Fundamentos conceptuales

### 4.1. Modelo Big Five/OCEAN

Se utiliza una representación de personalidad basada en cinco dimensiones:

| Dimensión | Descripción general |
|---|---|
| O — Openness | Apertura a nuevas experiencias. |
| C — Conscientiousness | Responsabilidad y organización. |
| E — Extraversion | Extraversión. |
| A — Agreeableness | Amabilidad y cooperación. |
| N — Neuroticism | Neuroticismo o tendencia a la inestabilidad emocional. |

En el prototipo, los rasgos pueden representarse mediante valores normalizados:

`R = [O, C, E, A, N]`, con cada componente en el intervalo `[0, 1]`.

Esta representación funciona como una abstracción computacional y no constituye una medición clínica de personalidad.

### 4.2. Arquetipos

MIPA utiliza configuraciones de comportamiento inspiradas en arquetipos junguianos, entre ellas:

- **THE_PROTECTOR:** estilo de respuesta orientado al apoyo y la protección.
- **THE_SAGE:** estilo orientado a la explicación, el análisis y la reflexión.
- **THE_JESTER:** estilo conversacional de carácter lúdico.
- **PROTOCOL_SAFETY:** configuración de respuesta prioritaria ante condiciones definidas por las reglas de seguridad.

Estas configuraciones son abstracciones de diseño. No representan categorías psicológicas diagnósticas ni equivalencias empíricamente demostradas entre los arquetipos y los rasgos OCEAN.

### 4.3. Lógica difusa

La arquitectura incorpora mecanismos de razonamiento difuso para representar transiciones graduales entre características y configuraciones de comportamiento.

A diferencia de una clasificación estrictamente binaria, una representación difusa permite expresar distintos grados de pertenencia a conjuntos definidos.

La correspondencia entre rasgos y arquetipos constituye una hipótesis de modelado que requiere evaluación y validación adicional.

### 4.4. Gobernanza ética

MIPA contempla restricciones explícitas de comportamiento y mecanismos de seguridad aplicados al proceso conversacional.

El diseño considera principios como:

- Transparencia y trazabilidad.
- Protección de información.
- Mitigación de sesgos.
- No patologización.
- Consistencia de comportamiento.
- Priorización de restricciones de seguridad.

Los mecanismos presentes en el prototipo son reglas y controles experimentales. **No constituyen una certificación de seguridad, garantía de ausencia de sesgos ni validación clínica.**

## 5. Tecnologías utilizadas

| Tecnología | Función |
|---|---|
| Python | Lenguaje principal de implementación. |
| FastAPI | API de comunicación y orquestación. |
| Streamlit | Interfaz de usuario. |
| Ollama | Ejecución local de modelos de lenguaje. |
| Transformers | Herramientas para arquitecturas de aprendizaje profundo. |
| Sentence Transformers | Representaciones semánticas mediante embeddings. |
| pysentimiento | Herramientas de análisis de lenguaje. |
| NumPy / SciPy | Operaciones numéricas y matemáticas. |
| pandas | Procesamiento de datos experimentales. |
| scikit-learn | Herramientas de aprendizaje automático. |
| statsmodels | Análisis estadístico. |
| Plotly | Visualización de información. |
| PyYAML | Lectura de configuración. |

Las versiones exactas de las dependencias Python se encuentran en `requirements.txt`.

## 6. Requisitos del entorno

El prototipo ha sido preparado y comprobado localmente con:

- **Python 3.12.8**
- Ollama instalado y disponible localmente.
- Un entorno virtual de Python.
- Las dependencias declaradas en `requirements.txt`.
- Modelos de lenguaje configurados en Ollama.

La disponibilidad de memoria RAM, almacenamiento y capacidad de procesamiento necesaria depende de los modelos utilizados.

La compatibilidad completa con otros sistemas operativos y configuraciones de hardware no ha sido validada exhaustivamente.

## 7. Instalación

### 7.1. Clonar el repositorio

```bash
git clone https://github.com/olavuenaz/MIPA---Proyecto---Version-2.git
cd MIPA---Proyecto---Version-2
```

### 7.2. Crear un entorno virtual

En Windows PowerShell:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

En Linux o macOS, con Python 3.12 instalado:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 7.3. Instalar dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Comprobar las dependencias instaladas:

```bash
python -m pip check
```

**Nota:** las versiones de `requirements.txt` corresponden al entorno de desarrollo utilizado. Su instalación y funcionamiento deben verificarse en cada plataforma de destino.

## 8. Configuración de modelos mediante Ollama

MIPA utiliza modelos locales registrados en Ollama.

La configuración experimental contempla los siguientes identificadores:

| Configuración | Modelo Ollama |
|---|---|
| THE_PROTECTOR | `mipa-protector` |
| THE_SAGE | `mipa-sabio` |
| THE_JESTER | `mipa-bufon` |
| PROTOCOL_SAFETY | `mipa-safety` |

Para consultar los modelos disponibles:

```bash
ollama list
```

La configuración general se encuentra en:

```text
config.yaml
```

Los identificadores de modelos deben corresponder a modelos efectivamente disponibles en la instalación local de Ollama.

### Adaptadores y pesos

El repositorio contiene directorios de adaptadores, archivos de configuración, tokenizadores y definiciones `Modelfile`.

**Los pesos completos de los modelos y adaptadores LoRA no están distribuidos como parte de este repositorio.**

Por esta razón, clonar el proyecto e instalar las dependencias Python no garantiza por sí solo la reproducción completa de la generación de respuestas.

La reconstrucción de los modelos personalizados requiere disponer de los artefactos compatibles, verificar sus licencias y seguir el procedimiento de preparación correspondiente.

No se debe asumir que los archivos `Modelfile` permiten reconstruir por sí solos los modelos entrenados.

## 9. Ejecución del prototipo

Los siguientes comandos deben ejecutarse desde la raíz del repositorio, con el entorno virtual activado.

### 9.1. Verificar la configuración

```bash
python -c "import yaml; print(yaml.safe_load(open('config.yaml', encoding='utf-8')).keys())"
```

### 9.2. Iniciar la API

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

La documentación interactiva de FastAPI estará disponible en:

`http://127.0.0.1:8000/docs`

### 9.3. Iniciar la interfaz

En otra terminal, con el entorno virtual activado:

```bash
streamlit run app_ui.py
```

La interfaz se abrirá en la dirección local indicada por Streamlit.

### 9.4. Funcionalidades de la API

El prototipo contempla las siguientes operaciones:

| Endpoint | Finalidad |
|---|---|
| `/chat` | Procesar una interacción conversacional. |
| `/update_profile` | Actualizar la configuración de un perfil. |
| `/reset_user_state` | Restablecer el estado de un usuario. |

Para conocer los métodos HTTP, parámetros y esquemas de solicitud vigentes, consultar la documentación OpenAPI generada por FastAPI.

**Verificación:** se ha comprobado la importación correcta de `main.py` en el entorno local de desarrollo. Esta comprobación no sustituye las pruebas funcionales de los endpoints ni la evaluación integral de las respuestas generadas.

## 10. Experimentación y evaluación

El repositorio incorpora herramientas para realizar experimentos controlados y analizar resultados.

### Archivos principales

| Archivo | Función |
|---|---|
| `run_baseline.py` | Ejecución de un procedimiento de referencia o baseline. |
| `run_experiment.py` | Ejecución de escenarios experimentales mediante solicitudes a la aplicación. |
| `analizar_para.py` | Procesamiento y análisis estadístico de resultados. |
| `test_set.jsonl` | Conjunto de casos de prueba. |
| `baseline_resultados.csv` | Resultados de referencia incluidos en el repositorio. |
| `experimento_resultados.csv` | Resultados experimentales incluidos en el repositorio. |
| `cuadros_tesis_mipa_completo.xlsx` | Cuadros y análisis asociados al trabajo académico. |

### Ejecución orientativa

Con los modelos disponibles y el entorno configurado:

```bash
python run_baseline.py
```

Para ejecutar el experimento, mantener activa la API y utilizar otra terminal:

```bash
python run_experiment.py
```

Para ejecutar el análisis:

```bash
python analizar_para.py
```

**Advertencia de reproducibilidad:** los nombres de archivos CSV incluidos en el repositorio pueden diferir de los nombres de salida esperados por los scripts. Es necesario revisar las rutas de entrada y salida antes de reproducir el análisis completo.

Asimismo, cualquier comparación estadística pareada requiere verificar la correspondencia de las observaciones y el tratamiento de respuestas faltantes o solicitudes fallidas.

### Alcance de los resultados

Las pruebas permiten estudiar comportamientos del prototipo en condiciones experimentales definidas.

Los resultados no deben interpretarse automáticamente como evidencia de personalidad humana, comprensión psicológica, seguridad generalizada o superioridad frente a otros sistemas conversacionales.

## 11. Datos sintéticos

Los perfiles de usuario utilizados en las pruebas son ficticios y fueron creados para la investigación.

El archivo:

```text
data_memory/user_profiles.json
```

incluye configuraciones sintéticas para estudiar diferentes condiciones experimentales.

Estos perfiles no representan personas reales ni constituyen registros clínicos.

Los archivos binarios generados por el almacenamiento de memoria local no forman parte de la versión de trabajo preparada para publicación. La exclusión de estos archivos en `.gitignore` no implica su eliminación automática de versiones históricas del repositorio.

## 12. Limitaciones conocidas

El prototipo presenta limitaciones técnicas y metodológicas que deben considerarse al interpretar sus resultados:

1. **Dependencia de modelos locales:** la ejecución completa requiere modelos compatibles registrados en Ollama.
2. **Distribución incompleta de pesos:** los adaptadores y modelos personalizados no pueden reconstruirse únicamente a partir de los archivos de configuración publicados.
3. **Integración parcial de memoria:** existen funcionalidades de memoria y recuperación, pero el flujo principal de conversación no incorpora actualmente contexto recuperado de manera completa.
4. **Integración parcial de componentes:** determinados módulos auxiliares de entrada y evaluación no forman parte del flujo principal de ejecución.
5. **Configuración de generación:** algunos parámetros declarados en YAML pueden no aplicarse directamente a la generación; la implementación actual utiliza parámetros establecidos en código.
6. **Seguridad experimental:** los mecanismos de protección son controles implementados para investigación, no garantías formales.
7. **Evaluación limitada:** las conclusiones dependen de los escenarios, perfiles y métricas empleados.
8. **Reproducibilidad estadística:** se requiere verificar la correspondencia entre observaciones y el manejo de fallos de ejecución.
9. **Compatibilidad multiplataforma:** no se ha realizado una validación exhaustiva en todas las plataformas y entornos.

Estas limitaciones forman parte del estado de desarrollo del prototipo y delimitan el alcance de las conclusiones de investigación.

## 13. Trabajo futuro

Entre las posibles líneas de desarrollo se encuentran:

- Integración completa de los componentes de memoria y recuperación contextual.
- Ampliación de los mecanismos de evaluación automática.
- Validación de la correspondencia conceptual entre rasgos OCEAN y arquetipos.
- Evaluación de consistencia de personalidad en conversaciones prolongadas.
- Estudios comparativos con modelos de referencia adicionales.
- Mejoras en la trazabilidad de decisiones.
- Pruebas de robustez, sesgos y comportamiento ante situaciones adversas.
- Empaquetado reproducible de modelos, configuraciones y experimentos.
- Ampliación de la validación interdisciplinaria.

## 14. Autoría y contexto académico

**Autor:** Eduardo Urbina Vergara  
**Institución:** Universidad Mayor, Chile  
**Carrera:** Ingeniería Civil en Computación e Informática  
**Proyecto:** Modelo de Inteligencia Artificial con Personalidad Adaptable (MIPA)

El proyecto se desarrolla en el contexto de una investigación académica que integra ingeniería de software, inteligencia artificial, procesamiento del lenguaje natural y fundamentos de psicología de la personalidad.

## 15. Licencias y componentes de terceros

Este repositorio incorpora o utiliza herramientas y componentes desarrollados por terceros, incluidos frameworks de Python, modelos de lenguaje, tokenizadores y utilidades de conversión.

Cada componente conserva las condiciones de licencia y atribución que correspondan.

Los scripts de conversión de modelos y los archivos derivados de herramientas externas requieren revisión de procedencia y licencias antes de asignar una licencia general al repositorio.

**La licencia de distribución del código original de MIPA está pendiente de definición.** La disponibilidad pública del repositorio no implica automáticamente autorización para reutilizar todos sus componentes sin restricciones.

## 16. Referencia académica

Para referirse al proyecto, se sugiere utilizar provisionalmente:

Urbina Vergara, E. (2026). *MIPA: Modelo de Inteligencia Artificial con Personalidad Adaptable* [Prototipo de investigación y código fuente]. GitHub.

Repositorio:

https://github.com/olavuenaz/MIPA---Proyecto---Version-2

Esta referencia es provisional y deberá actualizarse si se publica una versión formal del manuscrito, se asigna un DOI o se establece una versión académica definitiva.

---

**Estado del proyecto:** prototipo experimental de investigación.

MIPA se proporciona con fines académicos, educativos y de investigación. No constituye una herramienta de evaluación psicológica, diagnóstico clínico ni sustituto de asesoramiento profesional.