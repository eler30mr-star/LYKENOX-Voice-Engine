# Política de Identidad Propia y Distribución de LYKENOX

**ID de política:** LYX-POL-001  
**Versión:** 1.1  
**Estado:** OBLIGATORIA  
**Fecha de vigencia:** 4 de septiembre de 2026  
**Ámbito:** LYKENOX Voice Engine y todo trabajo de modelo, entrenamiento, inferencia, evaluación, empaquetado y distribución del producto LYKENOX

## 1. Objetivo

LYKENOX Voice Engine es un sistema de voz de identidad propia destinado a distribución independiente. Su identidad vocal, comportamiento del modelo, parámetros entrenados y ruta de inferencia de producción deben permanecer bajo control de LYKENOX y ser reproducibles a partir de datos, código, configuración y artefactos autorizados por el proyecto.

El proyecto no resolverá problemas de calidad, velocidad o ingeniería insertando inteligencia entrenada por terceros dentro del producto LYKENOX. La independencia de distribución y la propiedad de identidad prevalecen sobre la conveniencia, los resultados de benchmark o la rapidez de desarrollo.

Esta política distingue expresamente entre:

1. **implementación o dependencia del producto LYKENOX**, que debe permanecer propia; y
2. **herramientas externas offline de trabajo**, que pueden utilizarse fuera del proyecto para preparar, limpiar, inspeccionar, etiquetar, convertir o auditar datos, siempre que no se integren ni se conviertan en dependencia del producto y que sus términos permitan el uso de los resultados.

## 2. Política central

La pila de voz de producción utilizará exclusivamente **implementaciones de modelo mantenidas por LYKENOX y pesos de producción entrenados por LYKENOX**.

Ningún modelo preentrenado, checkpoint, activo de voz, servicio de inferencia o componente neuronal entrenado por terceros podrá introducirse, importarse, empaquetarse ni ejecutarse como parte de la ruta de identidad de voz de LYKENOX.

**Regla de bloqueo de implementación:** si producir la voz LYKENOX requiere que el repositorio, el entrenamiento propio del modelo, el paquete distribuido o la inferencia carguen, dependan de o invoquen pesos, identidad o inferencia entrenada fuera de LYKENOX, esa solución no es válida para este proyecto.

**Regla de herramienta externa offline:** una herramienta de terceros usada fuera del proyecto para transformar material de entrada no pasa a formar parte de LYKENOX por el solo hecho de que su resultado sea posteriormente usado como dato autorizado. Para ser válida, debe cumplir íntegramente la sección 6.

## 3. Alcance

Esta política se aplica a todo componente **implementado, incorporado o requerido por LYKENOX** que pueda influir en la voz generada o en su identidad, incluyendo:

- modelos texto-a-acústica y modelos acústicos;
- modelos de duración, alineamiento, F0/pitch y voicing;
- encoders de speaker o identidad;
- vocoders, renderers de waveform y módulos source-filter;
- postnets, denoisers, enhancers, codecs neuronales y postprocesamiento aprendido;
- checkpoints, pesos de inicialización, adapters y artefactos aprendidos;
- datasets de entrenamiento y validación, grabaciones de voz y datos sintéticos importados al proyecto;
- servicios de inferencia, endpoints remotos y APIs externas requeridos por el producto;
- artefactos exportados, empaquetados o distribuidos.

El uso manual u offline de una herramienta externa no constituye implementación en LYKENOX cuando esa herramienta permanece fuera del repositorio, build, runtime, entrenamiento del modelo LYKENOX e inferencia, y cuando únicamente se importa al proyecto su resultado como dato autorizado y validado.

## 4. Definiciones

### 4.1 Componente propiedad de LYKENOX

Componente cuya implementación se mantiene dentro del proyecto, cuyos pesos de producción son generados por el pipeline de entrenamiento de LYKENOX y cuya procedencia de datos y artefactos está autorizada para la distribución prevista.

### 4.2 Componente entrenado de tercero

Cualquier modelo, checkpoint, embedding, activo de voz, frontend aprendido, decoder aprendido, extractor de características aprendido o servicio de inferencia producido fuera del pipeline autorizado de LYKENOX.

Un componente entrenado de tercero **no puede formar parte de LYKENOX**. Su uso como herramienta externa offline solo es admisible bajo la sección 6 y no autoriza incorporarlo al repositorio, al entrenamiento propio del modelo, a la inferencia ni al paquete distribuido.

### 4.3 Dependencia genérica de infraestructura

Dependencia de software no identitaria utilizada por LYKENOX para cómputo, I/O de audio, operaciones tensoriales, empaquetado, pruebas, build o soporte del sistema operativo. Solo se considera infraestructura si no introduce identidad de voz ni pesos de voz preentrenados en el producto y cumple las reglas de distribución y licencia de esta política.

### 4.4 Herramienta externa offline

Software, servicio o aplicación de terceros utilizado **fuera de la implementación del proyecto** para preparar o inspeccionar material antes de que dicho material sea aceptado como dato LYKENOX. Puede ser tradicional o aprendido, siempre que:

- no se incluya ni se invoque desde el repositorio, build, runtime, entrenamiento del modelo LYKENOX o inferencia;
- no se distribuya con LYKENOX;
- LYKENOX no requiera esa herramienta para generar voz;
- no se importen sus pesos, embeddings, checkpoints, código identitario ni activaciones como parámetros del producto;
- no se use para inicializar, destilar, transferir o derivar pesos LYKENOX;
- el material de entrada tenga procedencia autorizada;
- la licencia, términos de servicio y derechos aplicables permitan utilizar el resultado con el fin previsto;
- el resultado sea auditado y aceptado como dato autorizado antes de entrar al corpus del proyecto.

### 4.5 Artefacto de datos autorizado

Audio, texto, etiqueta, metadata u otro dato que LYKENOX acepta como entrada de su propio pipeline después de verificar su procedencia, autorización, transformaciones relevantes y aptitud para el uso previsto. El hecho de haber sido transformado por una herramienta externa permitida no convierte esa herramienta en componente del producto.

## 5. Reglas no negociables

### Regla 1 - Prohibidos los pesos preentrenados de terceros dentro de LYKENOX

LYKENOX no cargará, importará, descargará, empaquetará, ajustará, destilará, inicializará desde ni dependerá de pesos preentrenados producidos por terceros **dentro de su código, entrenamiento de modelo, inferencia o distribución**.

La prohibición incluye checkpoints de model hubs, repositorios públicos, vendors, SaaS, releases de investigación, otros motores TTS y otros productos de voz cuando pretendan convertirse en componente o dependencia de LYKENOX.

El uso de una herramienta externa offline bajo la sección 6 no constituye carga de esos pesos por LYKENOX.

### Regla 2 - Prohibidos los componentes neuronales de voz de terceros en el producto

La ruta de producción no contendrá vocoders, TTS, acoustic models, speaker encoders, aligners, pitch models, enhancers, denoisers, codec models u otros componentes aprendidos de terceros que contribuyan a generar el habla.

Tampoco pueden convertirse en una dependencia necesaria del entrenamiento de los pesos de producción LYKENOX. Un artefacto de datos previamente preparado y aceptado no constituye tal dependencia.

### Regla 3 - Sin dependencia externa de inferencia

La inferencia de producción no requerirá API remota, modelo hospedado, endpoint de vendor, cloud TTS ni servicio externo para producir la voz LYKENOX.

El motor distribuido deberá poder generar su voz con sus propios artefactos LYKENOX y las dependencias genéricas declaradas.

### Regla 4 - Los pesos LYKENOX deben ser entrenados por LYKENOX

Todo archivo de parámetros aprendido que se distribuya deberá tener procedencia demostrable de una ejecución autorizada de entrenamiento LYKENOX.

No se permiten derivados de pesos de terceros mediante fine-tuning, transfer learning, distillation, adapters, LoRA, merging, conversión o mecanismos equivalentes.

Entrenar desde cero con datos autorizados que hayan sido limpiados o preparados externamente bajo la sección 6 **sí** se considera entrenamiento propio de LYKENOX.

### Regla 5 - Los datos de identidad deben ser propios o expresamente autorizados

Los datos que definan la identidad de voz deben ser propiedad del proyecto o estar expresamente autorizados para su uso y distribución.

No se mezclarán grabaciones de terceros, voces clonadas, datasets scrapeados, voces sintéticas de TTS externos ni activos de licencia no verificada.

Las transformaciones realizadas por herramientas externas offline deben quedar registradas en la procedencia del dataset cuando sean materialmente relevantes, y sus términos deben permitir el uso del resultado para entrenamiento y distribución del producto previsto.

### Regla 6 - Herramientas externas no son implementación del proyecto

LYKENOX puede usar herramientas externas offline para tareas auxiliares como limpieza/restauración de audio, edición, separación o reducción de ruido, inspección, etiquetado, transcripción asistida, conversión de formato, análisis o control de calidad, incluso cuando dichas herramientas utilicen modelos aprendidos de terceros, **siempre que permanezcan fuera de la implementación de LYKENOX**.

Para que este uso sea válido deben cumplirse todas estas condiciones:

1. La herramienta no se incorpora al repositorio ni al paquete distribuido.
2. El código LYKENOX no la carga, llama ni requiere para entrenar sus pesos de producción o generar voz.
3. Ningún peso, embedding, checkpoint o activación de la herramienta se importa como parámetro o inicialización de LYKENOX.
4. No existe fine-tuning, distillation, transfer learning, LoRA, merging ni derivación de pesos desde la herramienta hacia LYKENOX.
5. El material de origen es propio o autorizado.
6. La licencia/TOS de la herramienta permite el uso del resultado con el objetivo previsto, incluido uso comercial cuando corresponda.
7. El resultado se revisa y acepta como artefacto de datos LYKENOX; en audio de identidad debe comprobarse que conserva la identidad vocal y no introduce otras voces ni contenido ajeno.
8. Cuando la transformación sea relevante para procedencia o reproducibilidad, se documentarán herramienta, versión, configuración o procedimiento en el manifiesto del dataset.

Una herramienta externa que no cumpla cualquiera de estas condiciones queda `BLOCKED`.

### Regla 7 - La investigación puede informar; la implementación sigue siendo propia

Papers públicos, métodos matemáticos, estándares, herramientas e ideas arquitectónicas generales pueden estudiarse o utilizarse fuera del producto como referencia o apoyo.

La implementación de producción debe mantenerse dentro de LYKENOX y los pesos de producción deben entrenarse en LYKENOX. Estudiar o utilizar una herramienta externa no autoriza importar su checkpoint ni convertirla en componente del producto.

### Regla 8 - Prohibido maquillar fallos de calidad

No se usará normalización de ganancia, EQ, denoise, enhancement u otro postprocesamiento post-hoc **sobre la salida del candidato durante aceptación** para ocultar un modelo fallido.

Esta regla no prohíbe preparar o limpiar el dataset de entrada antes del entrenamiento. `CLEAN_V1` es preparación de datos; no puede utilizarse como excusa para procesar la salida del vocoder y hacer pasar un candidato defectuoso.

Cualquier postprocesamiento de producto deberá tener requisito explícito y, si forma parte de LYKENOX, cumplir la misma política de propiedad y distribución.

### Regla 9 - La duración predicha es un contrato independiente

La duración predicha no se modificará silenciosamente para hacer que un modelo parezca mejor. Cualquier cambio de duración requiere una decisión y gate independientes.

### Regla 10 - La calidad audible de utterances completas es la autoridad final

Las métricas pueden rechazar un candidato, pero no pueden aceptar por sí solas la calidad de voz.

La aceptación requiere escuchar utterances completas held-out generadas por la ruta prevista de producto. Crops cortos, smokes de trainability y mejoras numéricas aisladas no constituyen progreso de producto.

## 6. Herramientas externas offline e infraestructura permitida

### 6.1 Herramientas externas offline

Se permiten bajo la Regla 6. Su uso no convierte la herramienta, su modelo ni sus pesos en propiedad de LYKENOX y tampoco los integra al proyecto. LYKENOX conserva como artefacto propio/autorizado únicamente el material de entrada y el resultado cuya utilización esté permitida por los derechos y términos aplicables.

Para `CLEAN_V1`, una herramienta externa de limpieza puede utilizarse aunque sea aprendida, siempre que permanezca fuera del proyecto y se cumplan todas las condiciones de la Regla 6. El audio limpio debe superar validación auditiva para confirmar que conserva timbre, formantes, consonantes, ataques, respiraciones y demás rasgos de identidad útiles.

Cuando una contaminación esté solapada con la voz y la herramienta altere perceptiblemente la identidad, el fragmento debe rechazarse en vez de aceptarse por el mero hecho de estar más limpio.

### 6.2 Infraestructura genérica dentro del proyecto

Se permite software genérico de terceros dentro del proyecto cuando se cumplen simultáneamente estas condiciones:

1. No aporta identidad de voz preentrenada ni sustituye componentes aprendidos propios de producción.
2. Cumple una función general de runtime, cómputo numérico, audio, build, testing, empaquetado o sistema operativo.
3. Su licencia permite el modelo de distribución previsto para LYKENOX.
4. La dependencia y su licencia quedan documentadas para revisión de release.
5. Su uso no obliga a reemplazar la identidad entrenada propia por un modelo externo.

Ejemplos de categorías potencialmente válidas: frameworks tensoriales, FFT, lectores/escritores de archivos de audio, frameworks de testing y herramientas de empaquetado. La aceptación depende de su función y licencia, no de la marca.

## 7. Requisitos de distribución

Antes de aprobar una release que contenga modelos de voz, deberá verificarse:

- todo checkpoint distribuido posee procedencia LYKENOX;
- ningún modelo se descarga en el primer arranque desde un host de modelos de terceros;
- la ruta de producción no contiene IDs de modelos externos ni URLs de checkpoints preentrenados;
- la generación de voz no requiere servicios remotos de inferencia;
- la procedencia de los datos que definen identidad está documentada;
- las transformaciones externas materialmente relevantes del dataset tienen procedencia y derechos de uso documentados;
- las dependencias genéricas cuentan con revisión de licencia compatible con distribución;
- los hashes de modelos y checkpoints están registrados;
- la release puede reproducirse desde fuente, datos autorizados, configuración y artefactos LYKENOX aprobados;
- la aceptación auditiva se ha completado sobre utterances completas held-out.

No es requisito distribuir ni integrar las herramientas externas offline utilizadas para preparar datos. El artefacto de datos autorizado resultante constituye la entrada del pipeline LYKENOX.

## 8. Gates y enforcement de ingeniería

El repositorio mantendrá controles automatizados siempre que sea práctico. Como mínimo, los gates deberán rechazar o marcar:

- IDs de modelos preentrenados externos dentro de la ruta de voz de producción;
- cargas tipo `from_pretrained` para componentes de voz implementados en LYKENOX;
- descargas de checkpoints desde model hubs o proveedores externos realizadas por el producto o su pipeline de entrenamiento de pesos;
- llamadas nuevas a servicios remotos requeridas para generación de habla;
- incorporación al repositorio de denoisers, separadores, enhancers u otros modelos externos bajo la excusa de preprocessing;
- reactivación de familias perceptual o arquitectónicamente rechazadas sin decisión explícita y compatible con la política;
- entrenamiento persistente antes de superar gates de arquitectura, datos, exact-resume y aceptación.

Los checks de política son controles de rechazo. Superarlos no significa que un modelo tenga calidad perceptual aceptable.

## 9. Control de cambios de arquitectura y entrenamiento

No se autorizará una arquitectura nueva únicamente porque falló el candidato anterior. Antes de abrir una nueva línea deberá determinarse si el fallo pertenece a datos, conditioning, alignment, objetivo, procedimiento de entrenamiento, geometría del renderer o arquitectura.

El entrenamiento persistente requiere gate explícito. Los candidatos rechazados permanecen solo para reproducibilidad forense y no pueden convertirse silenciosamente en fuentes de inicialización de un modelo nuevo.

## 10. Excepciones y frontera de propiedad

**No existe excepción a la prohibición de componentes entrenados de terceros dentro de la identidad de voz, entrenamiento de pesos de producción o inferencia LYKENOX.**

El uso permitido de una herramienta externa offline según la Regla 6 **no es una excepción a esa prohibición**, porque la herramienta permanece fuera del proyecto y solo entrega un artefacto de datos autorizado.

El uso de una herramienta externa tampoco transfiere automáticamente derechos sobre sus modelos, software o resultados. Antes de aceptar un resultado como dato LYKENOX deben verificarse los derechos sobre el material de entrada y las condiciones aplicables al resultado.

La conveniencia, la velocidad de desarrollo, un benchmark favorable o la falta de una solución LYKENOX inmediata no justifican incorporar un componente de tercero al producto.

## 11. Regla de prioridad

Cuando existan objetivos de ingeniería en conflicto, se aplica el siguiente orden de autoridad:

1. Propiedad de identidad e independencia de distribución.
2. Corrección y reproducibilidad.
3. Calidad audible sobre utterances completas held-out.
4. Eficiencia de entrenamiento y conveniencia de implementación.

Un resultado técnicamente impresionante que introduzca una dependencia de modelo externo dentro de LYKENOX no es un resultado aceptable.

## 12. Autoridad de cambio de política

Esta política es obligatoria hasta que sea modificada mediante una decisión deliberada, explícita y versionada del proyecto.

Ningún experimento, ingeniero, agente automatizado, benchmark o tarea temporal puede debilitarla o saltarla implícitamente.

Toda modificación futura deberá identificar la regla exacta que cambia, su impacto de distribución, su impacto de propiedad y la razón de la modificación.

**Cambio v1.1:** se aclara que las herramientas externas offline usadas exclusivamente para preparar o inspeccionar datos no forman parte de la implementación LYKENOX cuando cumplen la Regla 6. Se mantiene sin excepción la prohibición de incorporar pesos, modelos o servicios aprendidos de terceros al producto, a sus pesos de producción o a su inferencia.

## 13. Criterio de cumplimiento y aprobación

Para considerar una versión del motor conforme a esta política deben existir evidencias trazables de propiedad de los pesos, procedencia de datos, ausencia de modelos externos en la ruta de producción y aceptación auditiva de utterances completas.

| Control | Requisito | Resultado requerido |
|---|---|---|
| Pesos | Procedencia LYKENOX demostrable | PASS |
| Modelos externos implementados | Ninguno en producto, entrenamiento de pesos o inferencia LYKENOX | 0 |
| Herramientas offline externas | Fuera del proyecto, derechos/TOS compatibles y transformaciones relevantes documentadas | PASS |
| Servicios remotos de producción | No requeridos para generar voz | 0 |
| Datos de identidad | Propios o expresamente autorizados | PASS |
| Licencias de infraestructura | Compatibles con distribución | PASS |
| Duración | Sin modificación silenciosa | PASS |
| Postprocesado de aceptación | No usado para ocultar fallos | PASS |
| Calidad | Utterances completas held-out aceptadas auditivamente | PASS |

---

**Declaración final de política:** LYKENOX es un proyecto de voz de identidad propia para distribución independiente. Su identidad vocal, sus implementaciones aprendidas, sus pesos de producción y su inferencia permanecerán bajo propiedad y entrenamiento de LYKENOX. Herramientas de terceros pueden asistir externamente en la preparación de datos sin formar parte del producto, siempre que se respeten procedencia, derechos, validación y la frontera de cero implementación de terceros dentro de LYKENOX.
