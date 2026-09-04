# Matriz viva de cumplimiento LYKENOX

**Política fuente:** `LYKENOX_IDENTITY_DISTRIBUTION_POLICY.md`  
**ID:** `LYX-POL-001`  
**Versión de matriz:** 1.0  
**Fecha de corte:** 2026-09-04  
**Estado global actual:** `COMPLIANT_DEVELOPMENT_WITH_PENDING_RELEASE_GATES`

Esta matriz traduce `LYX-POL-001` a controles operativos. No sustituye la política: cuando exista conflicto, manda `LYX-POL-001`.

## Estados

- `PASS`: hay evidencia suficiente para el estado actual del desarrollo.
- `PENDING`: no hay una violación conocida, pero falta evidencia o un gate que todavía no puede cerrarse.
- `BLOCKED`: la acción o componente viola la política o no está autorizado.
- `N/A`: no aplica al estado actual.

Un `PASS` de desarrollo no equivale a aprobación de release. Los controles marcados como de release deben repetirse sobre los artefactos finales.

## Matriz principal: reglas no negociables

| Regla | Control operativo | Estado | Evidencia actual | Condición para conservar/cerrar |
|---|---|---|---|---|
| R1 — Pesos preentrenados de terceros prohibidos | Ningún checkpoint, embedding o peso externo entra en entrenamiento, diagnóstico aprendido o producción | **PASS** | `speech_vocoder_active_source_decision.py`: `THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False`; los diagnósticos recientes usan modelos/artefactos LYKENOX y DSP/STFT genérico | Cualquier nueva dependencia aprendida exige rechazo salvo que sus pesos hayan sido entrenados por LYKENOX |
| R2 — Componentes neuronales de voz de terceros prohibidos | Vocoder, source model, aligner, denoiser, enhancer, codec o extractor aprendido que influya en identidad debe ser LYKENOX | **PASS** | Ruta activa y renderer implementados en el repositorio; no se introdujeron modelos externos durante los gates de fase/magnitud | `CLEAN_V1` no puede usar denoiser/separador neuronal preentrenado externo |
| R3 — Sin dependencia externa de inferencia | La generación no depende de API, cloud TTS ni servicio remoto | **PASS** | `REMOTE_MODEL_OR_TTS_SERVICE_AUTHORIZED = False` | Mantener inferencia local/reproducible con artefactos LYKENOX |
| R4 — Pesos LYKENOX entrenados por LYKENOX | Todo checkpoint distribuible debe tener procedencia de entrenamiento LYKENOX | **PASS** para artefactos actuales; **PENDING** para release final | El checkpoint forense actual proviene del pipeline LYKENOX; no está aceptado como producto | Registrar hashes, configuración, datos de origen y run final antes de release |
| R5 — Datos de identidad propios o autorizados | Grabaciones que definen identidad deben tener procedencia/autorización verificable | **PENDING** | El corpus de trabajo corresponde a la voz del proyecto, pero falta cerrar formalmente `CLEAN_V1` y su manifiesto de procedencia | Crear manifiesto de `RAW` y `CLEAN_V1`, registrar origen/autorización y rechazos/transformaciones |
| R6 — Sin excepción para probes | Diagnósticos no pueden introducir inteligencia entrenada prohibida | **PASS** | Griffin-Lim, STFT/ISTFT, phase swaps, magnitude swaps y demás gates son algoritmos genéricos/no aprendidos y artefactos LYKENOX | Todo nuevo probe debe declarar explícitamente modelos/pesos/servicios usados |
| R7 — Investigación puede informar; implementación propia | Se pueden estudiar métodos externos, pero producción/pesos siguen siendo LYKENOX | **PASS** | Documentos de investigación existen sin convertir checkpoints externos en ruta activa | Mantener separación entre referencia conceptual e implementación/pesos |
| R8 — Prohibido maquillar fallos de calidad | No usar gain/EQ/denoise/enhancement post-hoc para hacer pasar un candidato | **PASS** | `POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False`, `POSTHOC_EQ_AUTHORIZED = False`, `POSTHOC_DENOISING_AUTHORIZED = False`; los archivos `AUDITION` usan un único gain común solo para monitorización | El audio RAW de evaluación debe seguir disponible; cualquier procesado de producto requerirá gate independiente |
| R9 — Duración predicha es contrato independiente | No modificar duración silenciosamente para mejorar resultados | **PASS** | `PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False`; diagnósticos recientes no alteraron duración | Cualquier cambio futuro de duración requiere decisión/gate propio |
| R10 — Escucha held-out completa manda | Métricas pueden rechazar, pero no aceptar calidad final | **PASS** como proceso; **PENDING** como aceptación final | Las decisiones de fase, Griffin-Lim, dphase, group delay y spectral shape se tomaron por escucha; `METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False` | Repetir escucha completa held-out después de `CLEAN_V1` y sobre el modelo final de producción |

## Gates de distribución y release

| Gate de release | Estado | Evidencia / bloqueo actual | Criterio de cierre |
|---|---|---|---|
| Procedencia de todos los checkpoints finales | **PENDING** | El candidato actual es forense y no producto final | Manifiesto de entrenamiento + hash de cada checkpoint distribuido |
| Ausencia de IDs/URLs de modelos externos en producción | **PASS** actual / revalidar en release | No hay dependencia externa autorizada | Escaneo final de código/configuración/paquete |
| Inferencia sin servicio remoto | **PASS** actual / revalidar en release | Ruta local | Smoke de instalación offline |
| Procedencia de datos de identidad | **PENDING** | `CLEAN_V1` aún no construido | Manifiesto RAW→CLEAN con origen, transformaciones y segmentos rechazados |
| Licencias de infraestructura genérica | **PENDING** | No se ha cerrado auditoría de release | Inventario y revisión de licencias de runtime/audio/testing/package |
| Hashes de modelos y artefactos | **PENDING** | El modelo final aún no existe | Registro reproducible de hashes de release |
| Reproducibilidad desde fuente/datos/configuración | **PENDING** | Debe reconstruirse sobre `CLEAN_V1` | Run reproducible documentado y exact-resume verificado donde aplique |
| Aceptación auditiva held-out del modelo final | **PENDING** | El actual statistics source no está aceptado como producto | Utterances completas held-out aprobadas auditivamente sin post-hoc masking |

## Gate obligatorio `CLEAN_V1`

Estado: **PENDING / BLOQUEA NUEVO ENTRENAMIENTO DE FUENTE**.

Reglas de cumplimiento para `CLEAN_V1`:

1. Los archivos `RAW` originales son inmutables y nunca se sobrescriben.
2. Todo audio destinado a entrenamiento o a generar targets acústicos debe provenir de `CLEAN_V1` una vez activado ese corpus.
3. La limpieza puede usar edición/curación manual, DSP genérico no aprendido y algoritmos propios compatibles con la política.
4. Está **BLOCKED** usar denoisers, source separators, enhancers o modelos de restauración con pesos preentrenados de terceros, incluso solo para preparar datos.
5. Eventos externos claros —animales, motores, herramientas, viento fuerte, golpes, voces ajenas— se eliminan o se rechaza el fragmento cuando su separación dañaría la voz.
6. Deben preservarse timbre, formantes, consonantes, ataques, respiraciones y dinámica vocal útil.
7. Después de limpiar, se regeneran desde cero mel, F0, periodicidad, cepstrum, residual real, targets y caches. Los derivados del WAV sucio no se reutilizan.
8. La validación de `CLEAN_V1` es auditiva además de objetiva: no se acepta limpieza que introduzca metalización, burbujeo, pérdida de consonantes o cambio de identidad.
9. El denoise de salida del vocoder permanece prohibido durante gates de calidad; `CLEAN_V1` es preparación de datos, no parche de producto.

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
- **Denoiser externo preentrenado para limpiar dataset:** `BLOCKED` por R1/R2/R6.
- **Target phase copiada en inferencia:** `BLOCKED`; solo es oracle diagnóstico.
- **Post-hoc denoise/EQ/gain para aceptar vocoder:** `BLOCKED`.
- **Release final:** `BLOCKED` hasta cerrar los gates marcados `PENDING`.

## Próximo gate autorizado

`construct_and_audibly_validate_clean_v1_before_any_new_vocoder_training`
