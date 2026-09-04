# Matriz viva de cumplimiento LYKENOX

**Política fuente:** `LYKENOX_IDENTITY_DISTRIBUTION_POLICY.md`  
**ID:** `LYX-POL-001`  
**Versión de política:** 1.1  
**Versión de matriz:** 1.1  
**Fecha de corte:** 2026-09-04  
**Estado global actual:** `COMPLIANT_DEVELOPMENT_WITH_PENDING_RELEASE_GATES`

Esta matriz traduce `LYX-POL-001` a controles operativos. No sustituye la política: cuando exista conflicto, manda `LYX-POL-001`.

## Estados

- `PASS`: hay evidencia suficiente para el estado actual del desarrollo.
- `PENDING`: no hay una violación conocida, pero falta evidencia o un gate que todavía no puede cerrarse.
- `BLOCKED`: la acción o componente viola la política o no está autorizado.
- `N/A`: no aplica al estado actual.

Un `PASS` de desarrollo no equivale a aprobación de release. Los controles marcados como de release deben repetirse sobre los artefactos finales.

## Frontera de propiedad y herramientas externas

La política v1.1 separa claramente:

- **Dentro de LYKENOX:** cero implementación de modelos/pesos/servicios aprendidos de terceros en producto, entrenamiento de pesos o inferencia.
- **Fuera de LYKENOX:** se permiten herramientas offline de terceros para preparar, limpiar, inspeccionar, etiquetar, convertir o auditar datos, incluso si son aprendidas, siempre que no se integren al proyecto y que sus términos permitan usar el resultado.
- **Dato resultante:** puede entrar al corpus como artefacto autorizado LYKENOX después de validar procedencia, derechos y calidad.

Una herramienta externa offline no puede convertirse en dependencia silenciosa del repositorio, build, pipeline de entrenamiento de pesos, runtime o inferencia.

## Matriz principal: reglas no negociables

| Regla | Control operativo | Estado | Evidencia actual | Condición para conservar/cerrar |
|---|---|---|---|---|
| R1 — Sin pesos preentrenados de terceros dentro de LYKENOX | Ningún checkpoint, embedding o peso externo se carga/importa como parte del producto, entrenamiento de pesos o inferencia | **PASS** | `speech_vocoder_active_source_decision.py`: `THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False`; los modelos activos y checkpoints son LYKENOX | Herramientas offline externas pueden existir fuera del proyecto, pero sus pesos no pueden incorporarse a LYKENOX |
| R2 — Sin componentes neuronales de voz de terceros en el producto | Vocoder, source model, aligner, denoiser, enhancer, codec o extractor aprendido implementado en LYKENOX debe ser propio | **PASS** | Ruta activa y renderer implementados en el repositorio; no hay modelos externos en producción | `CLEAN_V1` puede prepararse externamente, pero ningún denoiser/separador externo se integra al repositorio o inferencia |
| R3 — Sin dependencia externa de inferencia | La generación no depende de API, cloud TTS ni servicio remoto | **PASS** | `REMOTE_MODEL_OR_TTS_SERVICE_AUTHORIZED = False` | Mantener inferencia local/reproducible con artefactos LYKENOX |
| R4 — Pesos LYKENOX entrenados por LYKENOX | Todo checkpoint distribuible debe tener procedencia de entrenamiento LYKENOX desde cero o desde artefactos propios autorizados | **PASS** para artefactos actuales; **PENDING** para release final | El checkpoint forense actual proviene del pipeline LYKENOX; no está aceptado como producto | Registrar hashes, configuración, datos de origen y run final antes de release |
| R5 — Datos de identidad propios o autorizados | Grabaciones que definen identidad deben tener procedencia/autorización verificable; transformaciones externas relevantes deben documentarse | **PENDING** | El corpus corresponde a la voz del proyecto, pero falta cerrar formalmente `CLEAN_V1` y su manifiesto | Crear manifiesto `RAW→CLEAN_V1`, registrar origen, herramienta/procedimiento, derechos y validación |
| R6 — Herramientas externas offline permitidas, implementación externa prohibida | Una herramienta externa puede preparar datos si permanece fuera del proyecto; no puede convertirse en dependencia del producto o del entrenamiento de pesos | **PASS** como regla; **PENDING** por herramienta concreta hasta revisar TOS/licencia | Política v1.1 define esta frontera explícitamente | Para cada herramienta usada: registrar nombre/versión/configuración cuando aplique, licencia/TOS, función y resultado validado |
| R7 — Investigación puede informar; implementación propia | Se pueden estudiar métodos/herramientas externos, pero producción/pesos siguen siendo LYKENOX | **PASS** | Documentos de investigación existen sin convertir checkpoints externos en ruta activa | Mantener separación entre referencia/apoyo y componentes del producto |
| R8 — Prohibido maquillar fallos de calidad | No usar gain/EQ/denoise/enhancement post-hoc sobre salida del candidato para hacerlo pasar | **PASS** | `POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False`, `POSTHOC_EQ_AUTHORIZED = False`, `POSTHOC_DENOISING_AUTHORIZED = False`; `AUDITION` usa gain común solo para escucha | Preparación de `CLEAN_V1` está permitida; denoise de salida del vocoder para aceptación sigue prohibido |
| R9 — Duración predicha es contrato independiente | No modificar duración silenciosamente para mejorar resultados | **PASS** | `PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False` | Cualquier cambio futuro requiere decisión/gate propio |
| R10 — Escucha held-out completa manda | Métricas pueden rechazar, pero no aceptar calidad final | **PASS** como proceso; **PENDING** como aceptación final | Decisiones recientes se tomaron por escucha; `METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False` | Repetir escucha completa held-out después de `CLEAN_V1` y sobre el modelo final |

## Reglas específicas para herramientas externas offline

Una herramienta externa offline es **PERMITTED** solo cuando se cumplen todas estas condiciones:

1. Se ejecuta fuera del repositorio/proyecto LYKENOX.
2. No se empaqueta ni distribuye con LYKENOX.
3. El código LYKENOX no la carga, llama ni necesita para generar voz.
4. No se importan a LYKENOX sus pesos, embeddings, checkpoints, activaciones o parámetros.
5. No se usa para fine-tuning, distillation, transfer learning, LoRA, merging o inicialización de pesos LYKENOX.
6. El material de entrada es propio o autorizado.
7. Sus términos/licencia permiten usar el resultado con el fin previsto, incluyendo uso comercial cuando corresponda.
8. El resultado se revisa antes de entrar al corpus; en audio de identidad debe conservar voz/timbre y no introducir otras voces o contenido ajeno.
9. Cuando sea relevante, se registra herramienta, versión, configuración/procedimiento y transformación en el manifiesto del dataset.

Ejemplos:

- `PERMITTED`: usar externamente una herramienta de limpieza de audio para producir un WAV limpio y después importar solo ese WAV validado a `CLEAN_V1`.
- `BLOCKED`: añadir el modelo de esa herramienta al repositorio o invocarlo desde scripts LYKENOX.
- `BLOCKED`: depender de una API externa durante inferencia de voz.
- `BLOCKED`: destilar o transferir conocimiento/pesos del modelo externo hacia un modelo LYKENOX.

## Gates de distribución y release

| Gate de release | Estado | Evidencia / bloqueo actual | Criterio de cierre |
|---|---|---|---|
| Procedencia de todos los checkpoints finales | **PENDING** | El candidato actual es forense y no producto final | Manifiesto de entrenamiento + hash de cada checkpoint distribuido |
| Ausencia de IDs/URLs/modelos externos implementados en producción | **PASS** actual / revalidar en release | No hay dependencia externa autorizada | Escaneo final de código/configuración/paquete |
| Inferencia sin servicio remoto | **PASS** actual / revalidar en release | Ruta local | Smoke de instalación offline |
| Procedencia de datos de identidad | **PENDING** | `CLEAN_V1` aún no construido | Manifiesto RAW→CLEAN con origen, herramientas externas usadas, derechos, transformaciones y segmentos rechazados |
| Derechos/TOS de herramientas externas offline usadas | **PENDING** si se usa alguna | Aún no se ha seleccionado herramienta concreta | Registrar evidencia de que el resultado puede usarse para entrenamiento/distribución previstos |
| Licencias de infraestructura genérica implementada | **PENDING** | No se ha cerrado auditoría de release | Inventario y revisión de licencias de runtime/audio/testing/package |
| Hashes de modelos y artefactos | **PENDING** | El modelo final aún no existe | Registro reproducible de hashes de release |
| Reproducibilidad desde fuente/datos/configuración | **PENDING** | Debe reconstruirse sobre `CLEAN_V1` | Run reproducible documentado y exact-resume verificado donde aplique |
| Aceptación auditiva held-out del modelo final | **PENDING** | El actual statistics source no está aceptado como producto | Utterances completas held-out aprobadas auditivamente sin post-hoc masking |

## Gate obligatorio `CLEAN_V1`

Estado: **PENDING / BLOQUEA NUEVO ENTRENAMIENTO DE FUENTE**.

Reglas de cumplimiento para `CLEAN_V1`:

1. Los archivos `RAW` originales son inmutables y nunca se sobrescriben.
2. Todo audio destinado a entrenamiento o a generar targets acústicos debe provenir de `CLEAN_V1` una vez activado ese corpus.
3. La limpieza puede usar edición/curación manual, DSP, algoritmos propios o herramientas externas offline de terceros, aprendidas o no, siempre que cumplan R6 y permanezcan fuera del proyecto.
4. Está **BLOCKED** incorporar al repositorio, pipeline de entrenamiento de pesos o inferencia cualquier denoiser, source separator, enhancer o modelo de restauración externo.
5. Eventos externos claros —animales, motores, herramientas, viento fuerte, golpes, voces ajenas— se eliminan o se rechaza el fragmento cuando su separación dañaría la voz.
6. Deben preservarse timbre, formantes, consonantes, ataques, respiraciones y dinámica vocal útil.
7. Después de limpiar, se regeneran desde cero mel, F0, periodicidad, cepstrum, residual real, targets y caches. Los derivados del WAV sucio no se reutilizan.
8. La validación de `CLEAN_V1` es auditiva además de objetiva: no se acepta limpieza que introduzca metalización, burbujeo, pérdida de consonantes o cambio de identidad.
9. El denoise de salida del vocoder permanece prohibido durante gates de calidad; `CLEAN_V1` es preparación de datos, no parche de producto.
10. Si se usa una herramienta externa, su uso queda documentado en el manifiesto de `CLEAN_V1`; la herramienta misma no se convierte en dependencia LYKENOX.

## Evidencia positiva congelada que no debe perderse

- El defecto de hop-grid/repetición de waveform heads quedó cerrado como fallo estructural de representación.
- Los oracles de residual real + cepstrum oracle + renderer fijo son GOLD y exoneran al renderer bajo el control probado.
- En `speech_0021`, candidate magnitude + target phase produce voz natural/correcta; la fase/coherencia temporal fue aislada como causa audible primaria.
- Griffin-Lim 64 es inteligible pero todavía robótico: no es solución final.
- Target temporal phase increments + candidate anchor producen una mejora grande; el anchor inicial por frecuencia importa.
- El smooth group-delay anchor empeoró y queda rechazado.
- En `speech_0022`, reference e identity roundtrip están limpios; el ruido tipo grinder aparece con la spectral shape candidata.
- `target_shape + candidate_level` elimina el grinder, mientras `candidate_shape + target_level` lo hace más evidente: la spectral shape candidata es el fallo de magnitud dominante observado en ese utterance.
- El diagnóstico de contaminación por transiente inicial está preservado, pero no bloquea el proyecto antes de `CLEAN_V1`.

Documento de milestone asociado: `docs/LYKENOX_VOCODER_POSITIVE_MILESTONE_2026-09-04.md`.

## Protocolo de actualización de esta matriz

La matriz debe actualizarse cuando ocurra cualquiera de estos eventos:

- se añada una dependencia que pueda influir en audio/identidad;
- se seleccione una herramienta externa para preparar datos;
- se cree o cambie un checkpoint candidato a producción;
- se modifique el pipeline de datos o limpieza;
- se autorice entrenamiento persistente nuevo;
- se cambie duración, renderer, postprocesado o ruta de inferencia;
- se cierre un gate de `CLEAN_V1`;
- se prepare una release.

Cada cambio debe incluir: estado anterior, estado nuevo, evidencia/commit, motivo y gate siguiente. Un cambio a `PASS` nunca puede basarse solo en una métrica cuando la política exige escucha o evidencia de procedencia.

## Bloqueos actuales

- **Nuevo entrenamiento de fuente:** `BLOCKED` hasta `CLEAN_V1` validado.
- **Nueva arquitectura de fuente:** `BLOCKED` mientras el gate activo sea reconstrucción de dataset/targets.
- **Implementar denoiser/separador externo dentro de LYKENOX:** `BLOCKED`.
- **Usar herramienta externa offline para limpiar datos:** `PERMITTED`, sujeto a R6 y revisión de derechos/TOS.
- **Target phase copiada en inferencia:** `BLOCKED`; solo es oracle diagnóstico.
- **Post-hoc denoise/EQ/gain para aceptar vocoder:** `BLOCKED`.
- **Release final:** `BLOCKED` hasta cerrar los gates marcados `PENDING`.

## Próximo gate autorizado

`construct_and_audibly_validate_clean_v1_before_any_new_vocoder_training`
