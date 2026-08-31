# Política de Identidad Propia y Distribución de LYKENOX

**ID de política:** LYX-POL-001  
**Versión:** 1.0  
**Estado:** OBLIGATORIA  
**Fecha de vigencia:** 31 de agosto de 2026  
**Ámbito:** LYKENOX Voice Engine y todo trabajo de modelo, entrenamiento, inferencia, evaluación, empaquetado y distribución

## 1. Objetivo

LYKENOX Voice Engine es un sistema de voz de identidad propia destinado a distribución independiente. Su identidad vocal, comportamiento del modelo, parámetros entrenados y ruta de inferencia de producción deben permanecer bajo control de LYKENOX y ser reproducibles a partir de datos, código, configuración y artefactos autorizados por el proyecto.

El proyecto no resolverá problemas de calidad, velocidad o ingeniería insertando inteligencia entrenada por terceros dentro del producto. La independencia de distribución y la propiedad de identidad prevalecen sobre la conveniencia, los resultados de benchmark o la rapidez de desarrollo.

## 2. Política central

La pila de voz de producción utilizará exclusivamente **implementaciones de modelo mantenidas por LYKENOX y pesos de producción entrenados por LYKENOX**.

Ningún modelo preentrenado, checkpoint, activo de voz, servicio de inferencia o componente neuronal entrenado por terceros podrá introducirse en la ruta de identidad de voz de LYKENOX, ya sea de forma permanente, temporal, como fallback, como sustituto de benchmark o como atajo de diagnóstico.

**Regla de bloqueo:** si una solución necesita pesos, identidad o inferencia entrenada fuera de LYKENOX, esa solución no es válida para este proyecto.

## 3. Alcance

Esta política se aplica a todo componente que pueda influir en la voz generada o en su identidad, incluyendo:

- modelos texto-a-acústica y modelos acústicos;
- modelos de duración, alineamiento, F0/pitch y voicing;
- encoders de speaker o identidad;
- vocoders, renderers de waveform y módulos source-filter;
- postnets, denoisers, enhancers, codecs neuronales y postprocesamiento aprendido;
- checkpoints, pesos de inicialización, adapters y artefactos aprendidos;
- datasets de entrenamiento y validación, grabaciones de voz y datos sintéticos;
- servicios de inferencia, endpoints remotos y APIs externas de generación de voz;
- artefactos exportados, empaquetados o distribuidos.

## 4. Definiciones

### 4.1 Componente propiedad de LYKENOX

Componente cuya implementación se mantiene dentro del proyecto, cuyos pesos de producción son generados por el pipeline de entrenamiento de LYKENOX y cuya procedencia de datos y artefactos está autorizada para la distribución prevista.

### 4.2 Componente entrenado de tercero

Cualquier modelo, checkpoint, embedding, activo de voz, frontend aprendido, decoder aprendido, extractor de características aprendido o servicio de inferencia producido fuera del pipeline autorizado de LYKENOX.

### 4.3 Dependencia genérica de infraestructura

Dependencia de software no identitaria utilizada para cómputo, I/O de audio, operaciones tensoriales, empaquetado, pruebas, build o soporte del sistema operativo. Solo se considera infraestructura si no contiene pesos o identidad de voz preentrenados y cumple las reglas de distribución y licencia de esta política.

## 5. Reglas no negociables

### Regla 1 - Prohibidos los pesos preentrenados de terceros

LYKENOX no cargará, importará, descargará, empaquetará, ajustará, destilará, inicializará desde ni dependerá de pesos preentrenados producidos por terceros.

La prohibición incluye checkpoints de model hubs, repositorios públicos, vendors, SaaS, releases de investigación, otros motores TTS y otros productos de voz.

### Regla 2 - Prohibidos los componentes neuronales de voz de terceros

La ruta de producción no contendrá vocoders, TTS, acoustic models, speaker encoders, aligners, pitch models, enhancers, denoisers, codec models u otros componentes aprendidos de terceros que contribuyan a generar el habla.

### Regla 3 - Sin dependencia externa de inferencia

La inferencia de producción no requerirá API remota, modelo hospedado, endpoint de vendor, cloud TTS ni servicio externo para producir la voz LYKENOX.

El motor distribuido deberá poder generar su voz con sus propios artefactos LYKENOX y las dependencias genéricas declaradas.

### Regla 4 - Los pesos LYKENOX deben ser entrenados por LYKENOX

Todo archivo de parámetros aprendido que se distribuya deberá tener procedencia demostrable de una ejecución autorizada de entrenamiento LYKENOX.

No se permiten derivados de pesos de terceros mediante fine-tuning, transfer learning, distillation, adapters, LoRA, merging, conversión o mecanismos equivalentes.

### Regla 5 - Los datos de identidad deben ser propios o expresamente autorizados

Los datos que definan la identidad de voz deben ser propiedad del proyecto o estar expresamente autorizados para su uso y distribución.

No se mezclarán grabaciones de terceros, voces clonadas, datasets scrapeados, voces sintéticas de TTS externos ni activos de licencia no verificada.

### Regla 6 - No existe excepción temporal para probes

Un componente entrenado prohibido no puede introducirse por ser llamado probe, benchmark, smoke test, diagnóstico, fallback, baseline comparativo o experimento temporal.

Los diagnósticos deberán construirse con artefactos propios o con herramientas genéricas no aprendidas.

### Regla 7 - La investigación puede informar; la implementación sigue siendo propia

Papers públicos, métodos matemáticos, estándares e ideas arquitectónicas generales pueden estudiarse como referencia.

La implementación de producción debe mantenerse dentro de LYKENOX y los pesos de producción deben entrenarse en LYKENOX. Estudiar una referencia no autoriza importar checkpoints ni insertar modelos entrenados externos.

### Regla 8 - Prohibido maquillar fallos de calidad

No se usará normalización de ganancia, EQ, denoise, enhancement u otro postprocesamiento post-hoc para ocultar un modelo fallido durante aceptación.

Cualquier postprocesamiento de producto deberá tener requisito explícito y cumplir la misma política de propiedad y distribución.

### Regla 9 - La duración predicha es un contrato independiente

La duración predicha no se modificará silenciosamente para hacer que un modelo parezca mejor. Cualquier cambio de duración requiere una decisión y gate independientes.

### Regla 10 - La calidad audible de utterances completas es la autoridad final

Las métricas pueden rechazar un candidato, pero no pueden aceptar por sí solas la calidad de voz.

La aceptación requiere escuchar utterances completas held-out generadas por la ruta prevista de producto. Crops cortos, smokes de trainability y mejoras numéricas aisladas no constituyen progreso de producto.

## 6. Infraestructura genérica permitida

Se permite software genérico de terceros únicamente cuando se cumplen simultáneamente estas condiciones:

1. No aporta identidad de voz preentrenada, pesos de modelo, embeddings ni comportamiento de habla aprendido.
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
- las dependencias genéricas cuentan con revisión de licencia compatible con distribución;
- los hashes de modelos y checkpoints están registrados;
- la release puede reproducirse desde fuente, datos, configuración y artefactos LYKENOX aprobados;
- la aceptación auditiva se ha completado sobre utterances completas held-out.

## 8. Gates y enforcement de ingeniería

El repositorio mantendrá controles automatizados siempre que sea práctico. Como mínimo, los gates deberán rechazar o marcar:

- IDs de modelos preentrenados externos dentro de la ruta de voz de producción;
- cargas tipo `from_pretrained` para componentes de voz de producción;
- descargas de checkpoints desde model hubs o proveedores externos;
- llamadas nuevas a servicios remotos de generación de habla;
- reactivación de familias perceptual o arquitectónicamente rechazadas sin decisión explícita y compatible con la política;
- entrenamiento persistente antes de superar gates de arquitectura, datos, exact-resume y aceptación.

Los checks de política son controles de rechazo. Superarlos no significa que un modelo tenga calidad perceptual aceptable.

## 9. Control de cambios de arquitectura y entrenamiento

No se autorizará una arquitectura nueva únicamente porque falló el candidato anterior. Antes de abrir una nueva línea deberá determinarse si el fallo pertenece a datos, conditioning, alignment, objetivo, procedimiento de entrenamiento, geometría del renderer o arquitectura.

El entrenamiento persistente requiere gate explícito. Los candidatos rechazados permanecen solo para reproducibilidad forense y no pueden convertirse silenciosamente en fuentes de inicialización de un modelo nuevo.

## 10. Excepciones

**No existe excepción a la prohibición de componentes entrenados de terceros dentro de la identidad de voz de producción LYKENOX.**

Cualquier excepción propuesta para infraestructura genérica debe documentarse, no debe contener inteligencia de voz entrenada y debe superar revisión de licencia y distribución antes de adoptarse.

La conveniencia, la velocidad de desarrollo, un benchmark favorable o la falta de una solución LYKENOX inmediata no justifican el incumplimiento.

## 11. Regla de prioridad

Cuando existan objetivos de ingeniería en conflicto, se aplica el siguiente orden de autoridad:

1. Propiedad de identidad e independencia de distribución.
2. Corrección y reproducibilidad.
3. Calidad audible sobre utterances completas held-out.
4. Eficiencia de entrenamiento y conveniencia de implementación.

Un resultado técnicamente impresionante que viole la propiedad o la independencia de distribución no es un resultado aceptable para LYKENOX.

## 12. Autoridad de cambio de política

Esta política es obligatoria hasta que sea modificada mediante una decisión deliberada, explícita y versionada del proyecto.

Ningún experimento, ingeniero, agente automatizado, benchmark o tarea temporal puede debilitarla o saltarla implícitamente.

Toda modificación futura deberá identificar la regla exacta que cambia, su impacto de distribución, su impacto de propiedad y la razón de la modificación.

## 13. Criterio de cumplimiento y aprobación

Para considerar una versión del motor conforme a esta política deben existir evidencias trazables de propiedad de los pesos, procedencia de datos, ausencia de modelos externos en la ruta de producción y aceptación auditiva de utterances completas.

| Control | Requisito | Resultado requerido |
|---|---|---|
| Pesos | Procedencia LYKENOX demostrable | PASS |
| Modelos externos | Ninguno en producción o diagnóstico aprendido | 0 |
| Servicios remotos | No requeridos para generar voz | 0 |
| Datos de identidad | Propios o expresamente autorizados | PASS |
| Licencias de infraestructura | Compatibles con distribución | PASS |
| Duración | Sin modificación silenciosa | PASS |
| Postprocesado | No usado para ocultar fallos | PASS |
| Calidad | Utterances completas held-out aceptadas auditivamente | PASS |

---

**Declaración final de política:** LYKENOX es un proyecto de voz de identidad propia para distribución independiente. Su identidad vocal y su stack aprendido de producción permanecerán bajo propiedad y entrenamiento de LYKENOX, auditables y distribuibles sin dependencias de modelos entrenados de terceros.
