# Derivación rigurosa de los modos de parametrización física
## Por indicador y por modo — todas las fórmulas y demostraciones paso a paso

---

## Notación y convenciones generales

A lo largo de este documento se usan las siguientes definiciones:

| Símbolo | Unidad | Significado |
|---|---|---|
| $f_s$ | Hz | Frecuencia de muestreo de la señal adquirida |
| $T_s = 1/f_s$ | s | Periodo de muestreo |
| $n_\text{rpm}$ | rev/min | Velocidad de rotación del husillo |
| $T_\text{rev} = 60/n_\text{rpm}$ | s | Periodo de una revolución del husillo |
| $f_r = 1/T_\text{rev}$ | Hz | Frecuencia de rotación |
| $f_\text{modal}$ | Hz | Frecuencia del modo de chatter dominante |
| $T_\text{modal} = 1/f_\text{modal}$ | s | Periodo del modo de chatter |
| $\lceil x \rceil$ | — | Función techo (menor entero $\geq x$) |
| $\lfloor x \rfloor$ | — | Función suelo (mayor entero $\leq x$) |

La función techo satisface: para todo $x \in \mathbb{R}$,

$$\lceil x \rceil = x + \varepsilon, \quad \varepsilon \in [0,1), \quad \varepsilon = 0 \iff x \in \mathbb{Z}.$$

En particular, si $x = N \cdot T \cdot f_s$ con $N, T, f_s > 0$, entonces el error de cuantización en tiempo es:

$$\Delta t = \frac{\lceil x \rceil - x}{f_s} = \frac{\varepsilon}{f_s} \in \left[0,\, \frac{1}{f_s}\right).$$

Para $f_s = 50\,000$ Hz esto representa un error máximo de $\Delta t < 20\,\mu\text{s}$, despreciable en el contexto de ventanas de análisis de decenas de milisegundos.

---

## 1. Indicador MaxEnt-SPRT

### 1.1 Funcionamiento interno y parámetros nativos

El indicador MaxEnt-SPRT opera sobre una representación discreta de la señal obtenida por decimación sincrónica OPR (*Once Per Revolution*). Dado un vector de tiempo $t$ y la señal $y(t)$ muestreada a $f_s$, la decimación OPR extrae una muestra cada vez que el husillo completa exactamente una revolución. El resultado es una secuencia

$$\mathbf{z} = \bigl(z_1, z_2, \ldots, z_M\bigr), \quad z_k = y\!\left(t_k\right),\quad t_k = k \cdot T_\text{rev},$$

cuya "frecuencia de muestreo efectiva" es $f_r = 1/T_\text{rev}$ [Hz]. Cada muestra $z_k$ corresponde exactamente a una revolución del husillo, de modo que el índice $k$ y el número de revolución son sinónimos.

Sobre la secuencia $\mathbf{z}$ se forman segmentos de $N_\text{seg}$ muestras consecutivas. El $i$-ésimo segmento es

$$\mathbf{z}^{(i)} = \bigl(z_{s_i}, z_{s_i+1}, \ldots, z_{s_i + N_\text{seg} - 1}\bigr), \quad s_i = 1 + (i-1)\cdot\text{step\_seg},$$

donde $\text{step\_seg} \in \{1, 2, \ldots, N_\text{seg}\}$ es el avance entre ventanas consecutivas. El número total de segmentos que se pueden extraer de una secuencia de longitud $M$ es

$$n_\text{segs} = \left\lfloor \frac{M - N_\text{seg}}{\text{step\_seg}} \right\rfloor + 1.$$

A cada segmento se le estima la distribución de entropía máxima gaussiana y se aplica el test SPRT. Los **parámetros nativos** del indicador son:

- $N_\text{seg} \in \mathbb{Z}^+$: número de muestras OPR por segmento (= número de revoluciones por ventana de análisis).
- $\text{rpm} > 0$: velocidad de rotación del husillo en rev/min; determina $T_\text{rev}$ y por tanto la escala temporal de cada segmento.
- $\text{step\_seg} \in \{1, \ldots, N_\text{seg}\}$: avance de la ventana deslizante en muestras OPR.

El problema de parametrización es que ninguno de estos tres valores tiene una interpretación inmediata en el espacio de pensamiento del investigador, que razona en términos de "¿cuántas revoluciones del husillo quiero que cubra cada ventana?" o "¿cuántos periodos del modo de chatter necesito por segmento para que la estimación de entropía sea fiable?". De ahí emergen los modos físicos.

---

### 1.2 Modo nativo (`param_mode = "native"`)

No existe transformación alguna. Los valores $N_\text{seg}$, $\text{rpm}$ y, opcionalmente, $\text{step\_seg}$ se leen directamente del diccionario de configuración y se pasan sin modificación al pipeline. Si $\text{step\_seg}$ no se especifica, el pipeline lo inicializa con $\text{step\_seg} = N_\text{seg}$ (ventanas no solapadas).

Se conserva este modo porque: (i) es el modo original sobre el que se construyeron los demás y facilita la comparación; (ii) permite reproducir exactamente configuraciones de la literatura donde los parámetros están expresados en unidades OPR.

---

### 1.3 Modo `by_revolution` — derivación completa

#### Parámetros de entrada

El usuario proporciona:

- $T_\text{rev} > 0$ [s]: periodo de una revolución del husillo.
- $N_\text{rev} \in \mathbb{Z}^+$: número de revoluciones deseado por segmento (notación del código: `N_rev_per_seg`).
- $\text{step\_rev} \in \mathbb{Z}^+$ (opcional): número de revoluciones de avance entre ventanas. Si se omite, $\text{step\_rev} = N_\text{rev}$.

#### Derivación de los parámetros nativos

**Paso 1 — Velocidad de rotación.**

$$\boxed{n_\text{rpm} = \frac{60}{T_\text{rev}}}$$

Esta es simplemente la relación definitoria entre periodo y velocidad angular. No hay error de conversión.

**Paso 2 — Número de muestras OPR por segmento.**

Por definición del decimador OPR, cada muestra de la secuencia $\mathbf{z}$ corresponde a exactamente una revolución del husillo. Por tanto la correspondencia es biyectiva:

$$\boxed{N_\text{seg} = N_\text{rev}}$$

No hay cuantización: la unidad física (revolución) es idéntica a la unidad nativa (muestra OPR). El cast a entero es sin pérdida porque $N_\text{rev}$ ya se exige entero.

**Paso 3 — Avance de la ventana.**

$$\boxed{\text{step\_seg} = \text{step\_rev}}$$

Misma identidad de unidades.

**Paso 4 — Duración temporal del segmento (verificación).**

$$t_\text{seg} = N_\text{seg} \cdot T_\text{rev} \quad [\text{s}]$$

Esta expresión no entra al pipeline; es una cantidad de trazabilidad que confirma cuántos segundos abarca cada ventana.

**Paso 5 — Restricciones de validez.**

$$T_\text{rev} > 0, \quad N_\text{rev} \geq 1, \quad 1 \leq \text{step\_rev} \leq N_\text{rev}.$$

La restricción $\text{step\_rev} \leq N_\text{rev}$ garantiza que la ventana avance al menos una muestra en cada paso y que el avance no supere el tamaño de la ventana.

**Paso 6 — Fracción de solapamiento.**

$$\rho = 1 - \frac{\text{step\_rev}}{N_\text{rev}} \in [0, 1).$$

- $\text{step\_rev} = N_\text{rev}$: $\rho = 0$ (sin solapamiento, ventanas adyacentes).
- $\text{step\_rev} = 1$: $\rho = 1 - 1/N_\text{rev}$ (solapamiento máximo).

**Ejemplo numérico** ($f_s = 50\,000$ Hz, $n_\text{rpm} = 12\,000$, $N_\text{rev} = 5$, $\text{step\_rev} = 1$):

$$T_\text{rev} = 60/12\,000 = 5 \times 10^{-3}\,\text{s}, \quad n_\text{rpm} = 12\,000\,\text{rev/min},$$
$$N_\text{seg} = 5, \quad \text{step\_seg} = 1, \quad t_\text{seg} = 5 \times 5\,\text{ms} = 25\,\text{ms}, \quad \rho = 1 - 1/5 = 0.80.$$

---

### 1.4 Modo `by_modal` — derivación completa

#### Motivación del problema

La restricción fundamental es que $N_\text{seg}$ debe ser un entero, porque la secuencia OPR es discreta y cada índice representa exactamente una revolución. Al mismo tiempo, el criterio de diseño de la ventana no es "cuántas revoluciones", sino "cuántos periodos del modo de chatter". Sea $N_m > 0$ el número deseado de periodos modales por ventana. La duración objetivo es

$$t_\text{target} = N_m \cdot T_\text{modal} \quad [\text{s}].$$

Expresando esta duración en número de revoluciones del husillo:

$$N_\text{rev}^\star = \frac{t_\text{target}}{T_\text{rev}} = \frac{N_m \cdot T_\text{modal}}{T_\text{rev}} = N_m \cdot \frac{f_r}{f_\text{modal}}.$$

En general $N_\text{rev}^\star \notin \mathbb{Z}$. Hay que mapear este valor real a un entero.

#### El problema del redondeo con `round`

La operación $N_\text{seg} = \text{round}(N_\text{rev}^\star)$ puede dar un entero *menor* que $N_\text{rev}^\star$ cuando la parte fraccionaria es $< 0.5$. En ese caso la ventana real sería más corta que la deseada y no capturaría $N_m$ periodos modales completos. Ejemplo:

$$n_\text{rpm} = 12\,000,\quad f_\text{modal} = 150\,\text{Hz}: \quad N_\text{rev}^\star = N_m \cdot \frac{200}{150} = N_m \cdot 1.333\ldots$$

Para $N_m = 5$: $N_\text{rev}^\star = 6.67$, $\text{round}(6.67) = 7$ (aceptable, da más de 5 periodos modales). Pero para $N_m = 3$: $N_\text{rev}^\star = 4.00$, $\text{round}(4.00) = 4$ (exacto en este caso). El problema aparece cuando la relación $f_r / f_\text{modal}$ produce fracciones justo por debajo de 0.5.

#### La solución adoptada: cambio de unidad base

En lugar de proyectar periodos modales al espacio de revoluciones del husillo, se redefine la unidad base del algoritmo OPR de modo que sea el **periodo modal** en vez del periodo revolucionario. Esto se implementa mediante:

$$\boxed{n_\text{rpm,modal} = \frac{60}{T_\text{modal}}}$$

$$\boxed{N_\text{seg} = \lfloor N_m \rfloor}$$

donde $N_m$ es el número entero de periodos modales deseado (en la práctica el usuario da $N_m$ como entero). El pipeline recibe $n_\text{rpm,modal}$ como si fuera la "velocidad de rotación", lo cual es una abstracción: internamente, cada "muestra OPR" corresponde ahora a un periodo modal de duración $T_\text{modal}$, no a una revolución del husillo.

#### Cuantización residual y su cálculo

La duración real del segmento, referida a muestras enteras de la señal a $f_s$, es:

$$N_\text{raw}^\text{seg} = \lceil N_\text{seg} \cdot T_\text{modal} \cdot f_s \rceil \quad \text{[muestras]},$$

de donde la duración real efectiva es

$$t_\text{real} = \frac{N_\text{raw}^\text{seg}}{f_s} = \frac{\lceil N_m \cdot T_\text{modal} \cdot f_s \rceil}{f_s}.$$

El error de cuantización es

$$\Delta t = t_\text{real} - t_\text{target} = \frac{\lceil N_m \cdot T_\text{modal} \cdot f_s \rceil - N_m \cdot T_\text{modal} \cdot f_s}{f_s} \in \left[0, \frac{1}{f_s}\right),$$

y el error porcentual relativo es

$$\delta_\% = \frac{\Delta t}{t_\text{target}} \times 100 = \frac{\lceil N_m \cdot T_\text{modal} \cdot f_s \rceil - N_m \cdot T_\text{modal} \cdot f_s}{N_m \cdot T_\text{modal} \cdot f_s} \times 100.$$

#### Parámetros de entrada

- $T_\text{rev} > 0$ [s]: periodo del husillo (para calcular $n_\text{rpm,modal}$ relativo a la unidad modal).
- $T_\text{modal} > 0$ [s]: periodo del modo de chatter.
- $N_m \in \mathbb{Z}^+$: número de periodos modales por segmento (notación del código: `N_modal_per_seg`).
- $\text{step\_modal} \in \mathbb{Z}^+$ (opcional): número de periodos modales de avance. Defecto: $\text{step\_modal} = N_m$.

#### Derivación de los parámetros nativos

**Paso 1 — "rpm modal"** (la unidad base es el periodo modal):

$$\boxed{n_\text{rpm,modal} = \frac{60}{T_\text{modal}}}$$

**Paso 2 — Número de "muestras" de la unidad base por segmento:**

$$\boxed{N_\text{seg} = N_m}$$

**Paso 3 — Avance:**

$$\boxed{\text{step\_seg} = \text{step\_modal}}$$

**Paso 4 — Duración objetivo y real:**

$$t_\text{target} = N_m \cdot T_\text{modal}, \quad t_\text{real} = \frac{\lceil N_m \cdot T_\text{modal} \cdot f_s \rceil}{f_s}.$$

**Paso 5 — Restricciones de validez:**

$$T_\text{rev} > 0,\quad T_\text{modal} > 0,\quad N_m \geq 1,\quad 1 \leq \text{step\_modal} \leq N_m.$$

**Ejemplo numérico** ($f_s = 50\,000$ Hz, $f_\text{modal} = 150$ Hz, $N_m = 5$, $\text{step\_modal} = 1$):

$$T_\text{modal} = 1/150 = 6.667\,\text{ms}, \quad n_\text{rpm,modal} = 60/0.006\overline{6} = 9\,000\,\text{rev/min (modal)},$$
$$t_\text{target} = 5 \times 6.667\,\text{ms} = 33.333\,\text{ms},$$
$$N_\text{raw}^\text{seg} = \lceil 5 \times 0.006\overline{6} \times 50\,000 \rceil = \lceil 1666.\overline{6} \rceil = 1667,$$
$$t_\text{real} = 1667/50\,000 = 33.34\,\text{ms}, \quad \Delta t = 0.007\,\text{ms}, \quad \delta_\% = 0.02\%.$$

---

### 1.5 Modo overlap — derivación completa

#### Definición y motivación

Sin solapamiento ($\text{step\_seg} = N_\text{seg}$), las ventanas son adyacentes y no comparten datos. La frecuencia de decisión del indicador es

$$f_\text{decisión}^{(0)} = \frac{f_r}{N_\text{seg}} = \frac{1}{N_\text{seg} \cdot T_\text{unit}} \quad [\text{Hz}],$$

donde $T_\text{unit}$ es $T_\text{rev}$ (modo `by_revolution`) o $T_\text{modal}$ (modo `by_modal`).

Para $N_\text{seg} = 5$, $T_\text{unit} = 5$ ms: $f_\text{decisión}^{(0)} = 1/(5 \times 0.005) = 40$ Hz, es decir, una decisión cada 25 ms.

Con solapamiento, $\text{step\_seg} < N_\text{seg}$ y la frecuencia de decisión aumenta:

$$\boxed{f_\text{decisión} = \frac{1}{\text{step\_seg} \cdot T_\text{unit}}}$$

Para $\text{step\_seg} = 1$: $f_\text{decisión} = 1/T_\text{unit} = f_r$ o $f_\text{modal}$ según el modo, es decir, una decisión por cada unidad temporal.

#### Fracción de solapamiento

La fracción de datos compartidos entre ventanas consecutivas $i$ e $i+1$ es

$$\rho = 1 - \frac{\text{step\_seg}}{N_\text{seg}} \in [0, 1).$$

Demostración: la ventana $i$ abarca los índices $[s_i, s_i + N_\text{seg})$; la ventana $i+1$ abarca $[s_i + \text{step\_seg},\, s_i + \text{step\_seg} + N_\text{seg})$. La región compartida tiene $N_\text{seg} - \text{step\_seg}$ muestras. Como fracción del total:

$$\rho = \frac{N_\text{seg} - \text{step\_seg}}{N_\text{seg}} = 1 - \frac{\text{step\_seg}}{N_\text{seg}}.$$

Casos límite:
- $\text{step\_seg} = N_\text{seg}$: $\rho = 0$ (sin solapamiento).
- $\text{step\_seg} = 1$: $\rho = (N_\text{seg}-1)/N_\text{seg}$ (máximo solapamiento posible manteniendo avance).

#### Número de segmentos extraíbles

Dado un vector OPR de longitud $M$ (muestras), el bucle deslizante produce segmentos mientras queden al menos $N_\text{seg}$ muestras desde el inicio de la ventana actual:

$$n_\text{segs} = \left\lfloor \frac{M - N_\text{seg}}{\text{step\_seg}} \right\rfloor + 1.$$

Demostración: la condición de que exista el segmento $i$ es $s_i + N_\text{seg} - 1 \leq M$, es decir, $1 + (i-1)\cdot\text{step\_seg} + N_\text{seg} - 1 \leq M$, de donde $(i-1) \leq (M - N_\text{seg})/\text{step\_seg}$, y el máximo $i$ es $\lfloor(M-N_\text{seg})/\text{step\_seg}\rfloor + 1$.

#### Conversión de las variables de solapamiento según el modo

En modo `by_revolution`:

$$\text{step\_seg} = \text{step\_rev}, \quad \text{step\_rev} \in \{1, \ldots, N_\text{rev}\}.$$

En modo `by_modal`:

$$\text{step\_seg} = \text{step\_modal}, \quad \text{step\_modal} \in \{1, \ldots, N_m\}.$$

En ambos casos la restricción es $1 \leq \text{step\_seg} \leq N_\text{seg}$.

---

### 1.6 Modo `segmentation = "raw"` — derivación completa

#### Limitación del esquema OPR y necesidad del modo raw

El decimador OPR actúa como un muestreador sincrónico con frecuencia efectiva $f_r = 1/T_\text{rev}$. Por el teorema de Nyquist-Shannon, la secuencia OPR solo puede representar correctamente frecuencias menores que $f_r/2$. Cualquier componente de $y(t)$ con frecuencia $f > f_r/2$ será aliasada o eliminada.

Para $n_\text{rpm} = 12\,000$: $f_r = 200$ Hz y $f_r/2 = 100$ Hz. Si $f_\text{modal} = 150$ Hz $> 100$ Hz, el modo de chatter cae fuera de la banda representable por la secuencia OPR: la decimación lo destruye antes de que el algoritmo lo procese.

El modo `raw` resuelve esto segmentando **directamente** la señal de alta frecuencia $y(t)$ a $f_s$ sin ningún paso previo de decimación. Cada segmento tiene $N_\text{raw}$ muestras de la señal original y cubre una duración de $N_\text{raw}/f_s$ segundos. La estimación de entropía MaxEnt opera sobre el histograma de amplitudes de este bloque raw.

#### Conversión en modo `by_revolution` + `raw`

El usuario especifica $N_\text{rev}$ revoluciones por segmento. La duración deseada es $t_\text{target} = N_\text{rev} \cdot T_\text{rev}$. El número de muestras de la señal raw que cubre esa duración es:

$$x = N_\text{rev} \cdot T_\text{rev} \cdot f_s = \frac{N_\text{rev} \cdot f_s}{f_r}.$$

Se aplica $\lceil \cdot \rceil$ para garantizar cobertura completa ($t_\text{real} \geq t_\text{target}$):

$$\boxed{N_\text{samples\_per\_seg} = \left\lceil N_\text{rev} \cdot T_\text{rev} \cdot f_s \right\rceil = \left\lceil \frac{N_\text{rev} \cdot f_s}{f_r} \right\rceil}$$

Para el paso:

$$\boxed{\text{step\_samples} = \left\lceil \text{step\_rev} \cdot T_\text{rev} \cdot f_s \right\rceil = \left\lceil \frac{\text{step\_rev} \cdot f_s}{f_r} \right\rceil}$$

Error de cuantización de la ventana:

$$\Delta t_\text{win} = \frac{\lceil N_\text{rev} \cdot T_\text{rev} \cdot f_s \rceil - N_\text{rev} \cdot T_\text{rev} \cdot f_s}{f_s} \in \left[0,\, \frac{1}{f_s}\right).$$

#### Conversión en modo `by_modal` + `raw`

El usuario especifica $N_m$ periodos modales por segmento. La duración deseada es $t_\text{target} = N_m \cdot T_\text{modal}$:

$$\boxed{N_\text{samples\_per\_seg} = \left\lceil N_m \cdot T_\text{modal} \cdot f_s \right\rceil}$$

$$\boxed{\text{step\_samples} = \left\lceil \text{step\_modal} \cdot T_\text{modal} \cdot f_s \right\rceil}$$

Error de cuantización:

$$\Delta t_\text{win} = \frac{\lceil N_m \cdot T_\text{modal} \cdot f_s \rceil - N_m \cdot T_\text{modal} \cdot f_s}{f_s} \in \left[0,\, \frac{1}{f_s}\right).$$

#### Por qué `ceil` y no `round` o `floor`

- `floor`: $\lfloor x \rfloor \leq x$, por lo que $t_\text{real} \leq t_\text{target}$. La ventana puede ser más corta que el objetivo, sin capturar el último ciclo completo deseado.
- `round`: $\text{round}(x) = \lfloor x + 0.5 \rfloor$, que puede ser $< x$ cuando $\{x\} < 0.5$. Misma objeción que `floor` en ese caso.
- `ceil`: $\lceil x \rceil \geq x$ siempre, por lo que $t_\text{real} \geq t_\text{target}$ garantizado. El exceso es estrictamente menor que $1/f_s$.

#### Compatibilidad con el modo overlap

El modo `raw` y el modo overlap son ortogonales y se combinan sin modificación de la lógica. La función `segment_signal_raw(y, t, N_\text{raw}, \text{step\_raw})` implementa el mismo bucle deslizante que `segment_opr`:

```
inicio = 0
mientras inicio + N_raw <= len(y):
    segmento[i] = y[inicio : inicio + N_raw]
    inicio += step_raw
```

La única diferencia respecto al modo OPR es que la unidad del índice es "muestra raw" en lugar de "muestra OPR".

**Ejemplo numérico** ($f_s = 50\,000$ Hz, $f_r = 200$ Hz, $N_\text{rev} = 5$, $\text{step\_rev} = 1$):

$$N_\text{samples\_per\_seg} = \left\lceil 5 \times \frac{50\,000}{200} \right\rceil = \lceil 1250 \rceil = 1250,$$
$$\text{step\_samples} = \left\lceil 1 \times 250 \right\rceil = 250, \quad \rho = 1 - 250/1250 = 0.80.$$

**Ejemplo numérico** ($f_s = 50\,000$ Hz, $f_\text{modal} = 150$ Hz, $N_m = 5$, $\text{step\_modal} = 1$):

$$N_\text{samples\_per\_seg} = \left\lceil 5 \times \frac{1}{150} \times 50\,000 \right\rceil = \left\lceil 1666.\overline{6} \right\rceil = 1667,$$
$$\text{step\_samples} = \left\lceil \frac{50\,000}{150} \right\rceil = \lceil 333.\overline{3} \rceil = 334.$$

---

## 2. Indicador RMS-CV

### 2.1 Funcionamiento interno y parámetros nativos

El indicador RMS-CV opera en dos etapas sucesivas sobre la señal $y(t)$ muestreada a $f_s$.

**Etapa 1 — Secuencia de RMS.** La señal se divide en ventanas deslizantes de $W$ muestras con solapamiento $\rho \in [0,1)$. El paso entre ventanas es $s_W = \lfloor W(1-\rho) \rfloor$ muestras. Para cada ventana $j$ se calcula

$$\text{RMS}_j = \sqrt{\frac{1}{W}\sum_{k=0}^{W-1} y_{j\cdot s_W + k}^2}.$$

**Etapa 2 — Monitor CV en línea.** Sobre la secuencia de valores $\{\text{RMS}_j\}$ se aplica un monitor de ventana deslizante de $m$ frames. En cada paso se calcula

$$\mu_n = \frac{1}{n}\sum_{i=1}^{n}\text{RMS}_i, \quad \sigma_n = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(\text{RMS}_i - \mu_n)^2}, \quad \text{CV}_n = \frac{\sigma_n}{\mu_n + \varepsilon},$$

donde $n = \min(j, m)$ es el número de frames activos en la ventana. Se detecta chatter cuando $\text{CV}_n \geq \tau_\text{CV}$.

Los **parámetros nativos** son:

- $W \in \mathbb{Z}^+$: muestras por ventana RMS.
- $\rho \in [0,1)$: fracción de solapamiento entre ventanas RMS.
- $m \in \mathbb{Z}^+$: tamaño de la ventana del monitor CV (número de frames RMS).

La separación temporal entre frames RMS consecutivos (paso efectivo) es

$$\Delta t_\text{RMS} = \frac{W}{f_s} \cdot (1 - \rho) \quad [\text{s}].$$

---

### 2.2 Modo nativo (`param_mode = "native"`)

Los valores $W$, $\rho$ y $m$ se pasan directamente al pipeline. No hay transformación.

---

### 2.3 Modo `by_revolution`, `n_max_mode = "frames"` — derivación completa

#### Parámetros de entrada

- $T_\text{unit} = T_\text{rev}$ [s] (modo `by_revolution`) o $T_\text{unit} = T_\text{modal}$ [s] (modo `by_modal`).
- $N_\text{win} > 0$: número de unidades temporales por ventana RMS (`N_rev_window` o `N_modal_window`).
- $\text{step} \in (0, N_\text{win}]$: número de unidades de avance (`step_rev` o `step_modal`).
- $m \in \mathbb{Z}^+$: número de frames RMS en la ventana CV (`n_max_rev` o `n_max_modal`).

**Restricciones:**

$$T_\text{unit} > 0, \quad N_\text{win} \geq 1, \quad 0 < \text{step} \leq N_\text{win}, \quad m \geq 1.$$

#### Derivación de los parámetros nativos

**Paso 1 — Muestras por ventana RMS.**

La duración exacta de la ventana es $t_\text{win}^\star = N_\text{win} \cdot T_\text{unit}$ [s]. El número entero de muestras que la cubre sin quedar corto es

$$\boxed{W = \left\lceil N_\text{win} \cdot T_\text{unit} \cdot f_s \right\rceil}$$

**Paso 2 — Duración real de la ventana.**

$$t_\text{win} = \frac{W}{f_s} = \frac{\lceil N_\text{win} \cdot T_\text{unit} \cdot f_s \rceil}{f_s} \geq t_\text{win}^\star.$$

El error de cuantización es $\Delta t_\text{win} = t_\text{win} - t_\text{win}^\star \in [0, 1/f_s)$.

**Paso 3 — Fracción de solapamiento entre ventanas RMS.**

$$\boxed{\rho = 1 - \frac{\text{step}}{N_\text{win}}}$$

Demostración: el avance en unidades temporales es $\text{step} \cdot T_\text{unit}$ y la ventana cubre $N_\text{win} \cdot T_\text{unit}$. La fracción solapada es

$$\rho = \frac{N_\text{win} - \text{step}}{N_\text{win}} = 1 - \frac{\text{step}}{N_\text{win}}.$$

**Paso 4 — Paso temporal entre frames RMS.**

El avance exacto en tiempo es $\Delta t^\star = \text{step} \cdot T_\text{unit}$. El paso real, usando $t_\text{win}$ y $\rho$:

$$\Delta t_\text{RMS} = t_\text{win} \cdot (1 - \rho) = t_\text{win} \cdot \frac{\text{step}}{N_\text{win}}.$$

Obsérvese que $\Delta t_\text{RMS} \neq \Delta t^\star$ en general (difieren en $O(1/f_s)$) porque $t_\text{win}$ está cuantizada.

**Paso 5 — Número de frames CV.**

$$\boxed{m = m_\text{input}}$$

Se pasa directamente sin conversión.

**Paso 6 — Duración total de la ventana CV.**

Exacta (deseada):

$$T_\text{CV}^\star = t_\text{win}^\star + (m-1) \cdot \Delta t^\star = N_\text{win} \cdot T_\text{unit} + (m-1) \cdot \text{step} \cdot T_\text{unit} = \bigl[N_\text{win} + (m-1)\cdot\text{step}\bigr] \cdot T_\text{unit}.$$

Real (efectiva):

$$T_\text{CV} = t_\text{win} + (m-1) \cdot \Delta t_\text{RMS}.$$

Expresada en unidades temporales: $K_\text{CV} = T_\text{CV} / T_\text{unit}$ y $K_\text{CV}^\star = N_\text{win} + (m-1)\cdot\text{step}$.

**Ejemplo numérico** ($f_s = 50\,000$ Hz, modo `by_revolution`, $T_\text{rev} = 5$ ms, $N_\text{win} = 5$, $\text{step} = 1$, $m = 28$):

$$W = \lceil 5 \times 0.005 \times 50\,000 \rceil = \lceil 1250 \rceil = 1250,$$
$$\rho = 1 - 1/5 = 0.80, \quad \Delta t_\text{RMS} = (1250/50\,000)\times(1/5) = 0.005\,\text{s} = 5\,\text{ms},$$
$$K_\text{CV}^\star = 5 + 27 \times 1 = 32\,\text{rev}, \quad T_\text{CV}^\star = 32 \times 5\,\text{ms} = 160\,\text{ms}.$$

---

### 2.4 Modo `by_revolution`, `n_max_mode = "total_window"` — derivación completa

#### Planteamiento

El usuario no especifica $m$ directamente sino el span total deseado de la ventana CV, $K$ (en las mismas unidades que $N_\text{win}$ y $\text{step}$). Se necesita despejar $m$ en función de $K$.

#### Derivación directa: span como función de $m$

La secuencia de $m$ frames RMS, cada uno de ancho $N_\text{win}$ unidades y con avance $\text{step}$ unidades entre inicios consecutivos, cubre la región temporal desde el inicio del frame 1 hasta el final del frame $m$:

- Inicio del frame 1: posición $0$.
- Fin del frame 1: posición $N_\text{win}$.
- Inicio del frame $m$: posición $(m-1)\cdot\text{step}$.
- Fin del frame $m$: posición $(m-1)\cdot\text{step} + N_\text{win}$.

El span total desde el inicio del frame 1 hasta el final del frame $m$ es por tanto

$$K(m) = (m-1)\cdot\text{step} + N_\text{win}.$$

#### Inversión: $m$ como función de $K$

Despejando $m$ de la ecuación anterior:

$$K - N_\text{win} = (m-1)\cdot\text{step}$$
$$m - 1 = \frac{K - N_\text{win}}{\text{step}}$$
$$m = \frac{K - N_\text{win}}{\text{step}} + 1.$$

Si el valor de $K$ especificado por el usuario no produce un $m$ entero exacto, se necesita el menor entero $m \in \mathbb{Z}^+$ tal que $K(m) \geq K$:

$$K(m) \geq K \iff (m-1)\cdot\text{step} + N_\text{win} \geq K \iff m \geq \frac{K-N_\text{win}}{\text{step}} + 1.$$

El menor entero que satisface esta desigualdad es

$$\boxed{m = \left\lceil \frac{K - N_\text{win}}{\text{step}} + 1 \right\rceil}$$

#### Span real resultante

El span real, usando el $m$ entero calculado, es

$$K_\text{real} = N_\text{win} + (m-1)\cdot\text{step} \geq K.$$

El exceso es

$$K_\text{real} - K = \left(\left\lceil \frac{K-N_\text{win}}{\text{step}} + 1 \right\rceil - 1\right)\cdot\text{step} + N_\text{win} - K \in [0, \text{step}).$$

Demostración: sea $q = (K-N_\text{win})/\text{step}$ y $m = \lceil q+1\rceil = \lfloor q\rfloor + 1 + \mathbf{1}[q \notin \mathbb{Z}]$. Entonces $(m-1)\cdot\text{step} = \lceil q \rceil \cdot \text{step}$, y $K_\text{real} - K = (\lceil q\rceil - q)\cdot\text{step} \in [0, \text{step})$.

#### Restricción de validez

Se necesita $m \geq 2$ (el monitor CV no tiene sentido con un solo frame). De $m \geq 2$:

$$\left\lceil \frac{K-N_\text{win}}{\text{step}} + 1 \right\rceil \geq 2 \iff \frac{K-N_\text{win}}{\text{step}} + 1 > 1 \iff K > N_\text{win}.$$

Por eso el resolver exige $K > N_\text{win}$.

**Ejemplo numérico** ($N_\text{win} = 16$, $\text{step} = 16$, $K = 448$):

$$m = \left\lceil \frac{448 - 16}{16} + 1 \right\rceil = \lceil 27 + 1 \rceil = 28. \quad K_\text{real} = 16 + 27\times16 = 448. \quad\text{(exacto)}$$

**Ejemplo numérico** ($N_\text{win} = 5$, $\text{step} = 3$, $K = 20$):

$$\frac{20-5}{3} + 1 = 5 + 1 = 6 \Rightarrow m = 6. \quad K_\text{real} = 5 + 5\times3 = 20. \quad\text{(exacto)}$$

**Ejemplo con exceso** ($N_\text{win} = 5$, $\text{step} = 3$, $K = 19$):

$$\frac{19-5}{3} + 1 = 4.6\overline{6} + 1 = 5.6\overline{6} \Rightarrow m = 6. \quad K_\text{real} = 20 > 19.$$

---

### 2.5 Modo `by_modal`, `n_max_mode = "frames"` — derivación completa

Estructuralmente idéntico a la Sección 2.3, con $T_\text{unit} = T_\text{modal}$ en todas las fórmulas. Los parámetros de entrada son `T_modal`, `N_modal_window`, `step_modal`, `n_max_modal`. Las expresiones son:

$$W = \left\lceil N_\text{modal\_win} \cdot T_\text{modal} \cdot f_s \right\rceil, \quad \rho = 1 - \frac{\text{step\_modal}}{N_\text{modal\_win}}, \quad m = m_\text{modal}.$$

La diferencia conceptual respecto al modo `by_revolution` del MaxEnt-SPRT es importante: aquí no existe la restricción de que $N_\text{win}$ deba ser un número entero de revoluciones del husillo, porque el pipeline RMS-CV opera sobre muestras de la señal original a $f_s$ y no sobre muestras OPR. Por tanto, la cuantización se produce solo en la conversión muestras $= \lceil N_\text{win}\cdot T_\text{unit}\cdot f_s \rceil$, sin conflicto entre dos relojes (husillo y modo).

**Ejemplo numérico** ($f_s = 50\,000$ Hz, $f_\text{modal} = 150$ Hz, $N_\text{modal\_win} = 5$, $\text{step\_modal} = 1$, $m = 10$):

$$T_\text{modal} = 1/150\,\text{s}, \quad W = \left\lceil 5 \times \frac{1}{150} \times 50\,000 \right\rceil = \left\lceil 1666.\overline{6} \right\rceil = 1667,$$
$$\rho = 1 - 1/5 = 0.80, \quad t_\text{win} = 1667/50\,000 = 33.34\,\text{ms}.$$

---

## 3. Indicador SSQ-STFT + SVD

### 3.1 Funcionamiento interno y parámetros nativos

El indicador SSQ-STFT+SVD construye una representación tiempo-frecuencia $\mathbf{S} \in \mathbb{R}^{F \times L}$ (donde $F$ es el número de bins de frecuencia y $L$ el número de columnas temporales) mediante STFT o su variante sincro-exprimida SSQ. Los parámetros que controlan la resolución de esta representación son:

- $w$ [ms]: duración de la ventana de análisis de la STFT.
- $h$ [ms]: salto (*hop*) entre ventanas STFT consecutivas.

El número de muestras correspondientes (calculado internamente por el pipeline) es

$$n_\text{win} = \left\lceil w \times 10^{-3} \times f_s \right\rceil, \quad n_\text{hop} = \left\lceil h \times 10^{-3} \times f_s \right\rceil.$$

Sobre la matriz $\mathbf{S}$, una ventana deslizante de $A_i$ columnas consecutivas es extraída. A cada submatriz $\mathbf{S}_{:,\,j:j+A_i}$ se aplica SVD:

$$\mathbf{S}_{:,\,j:j+A_i} = \mathbf{U}\,\boldsymbol{\Sigma}\,\mathbf{V}^\top,$$

y el primer valor singular $d_1^{(j)} = \sigma_1$ es la característica de detección. Se declara chatter cuando $d_1$ supera un umbral estadístico calculado sobre el periodo estable de la señal.

Los **parámetros nativos** son:

- $w$ [ms]: duración de la ventana STFT.
- $h$ [ms]: salto entre columnas STFT.
- $A_i \in \mathbb{Z}^+$: número de columnas STFT en la ventana SVD.

Restricción: $0 \leq h \leq w$ (el salto no puede superar la ventana).

---

### 3.2 Modo nativo (`param_mode = "native"`)

Los tres parámetros se pasan directamente. No hay transformación.

---

### 3.3 Modo `by_revolution`, `Ai_length_mode = "frames"` — derivación completa

#### Parámetros de entrada

- $T_\text{unit}$: $T_\text{rev}$ o $T_\text{modal}$ según el sub-modo.
- $N_\text{win} > 0$: número de unidades por ventana STFT.
- $\text{step} \in (0, N_\text{win}]$: número de unidades de avance (*hop*).
- $A_i \in \mathbb{Z}^+$: número de columnas STFT en la ventana SVD (dato directo).

#### Derivación de los parámetros nativos

**Paso 1 — Duración de la ventana STFT (exacta en ms):**

$$\boxed{w = N_\text{win} \cdot T_\text{unit} \times 10^3 \quad [\text{ms}]}$$

**Paso 2 — Salto STFT (exacto en ms):**

$$\boxed{h = \text{step} \cdot T_\text{unit} \times 10^3 \quad [\text{ms}]}$$

A diferencia del RMS-CV, el resolver SSQ **no aplica `ceil`** en este punto. Los valores $w$ y $h$ son flotantes exactos. La cuantización a muestras la realiza el pipeline internamente:

$$n_\text{win} = \lceil w \times 10^{-3} \times f_s \rceil = \lceil N_\text{win} \cdot T_\text{unit} \cdot f_s \rceil,$$
$$n_\text{hop} = \lceil h \times 10^{-3} \times f_s \rceil = \lceil \text{step} \cdot T_\text{unit} \cdot f_s \rceil.$$

Sin embargo, para el cálculo del `trace`, el resolver sí computa los valores efectivos:

$$w_\text{ef} = \frac{n_\text{win}}{f_s} \times 10^3, \quad h_\text{ef} = \frac{n_\text{hop}}{f_s} \times 10^3.$$

Y pasa $w_\text{ef}$ y $h_\text{ef}$ (no $w$ y $h$) como `native_params`, de modo que cuando el pipeline vuelve a aplicar `ceil` obtiene el mismo $n_\text{win}$ (la operación $\lceil \lceil x \rceil \rceil = \lceil x \rceil$ es idempotente).

**Paso 3 — Validación de la restricción $h \leq w$:**

$$h \leq w \iff \text{step} \cdot T_\text{unit} \leq N_\text{win} \cdot T_\text{unit} \iff \text{step} \leq N_\text{win}.$$

Esta restricción ya está garantizada por $\text{step} \in (0, N_\text{win}]$.

**Paso 4 — Número de columnas SVD:**

$$\boxed{A_i = A_{i,\text{input}}}$$

**Paso 5 — Span total de la ventana SVD (trazabilidad).**

Exacto:

$$T_\text{SVD}^\star = w + (A_i - 1)\cdot h = \bigl[N_\text{win} + (A_i-1)\cdot\text{step}\bigr]\cdot T_\text{unit}\times 10^3 \quad [\text{ms}]$$

Efectivo:

$$T_\text{SVD} = w_\text{ef} + (A_i-1)\cdot h_\text{ef} = \frac{n_\text{win} + (A_i-1)\cdot n_\text{hop}}{f_s} \times 10^3 \quad [\text{ms}]$$

**Ejemplo numérico** ($f_s = 50\,000$ Hz, $T_\text{rev} = 5$ ms, $N_\text{win} = 5$, $\text{step} = 1$, $A_i = 3$):

$$w = 5\times5\,\text{ms} = 25\,\text{ms}, \quad h = 1\times5\,\text{ms} = 5\,\text{ms},$$
$$n_\text{win} = \lceil 25\times10^{-3}\times50\,000 \rceil = \lceil 1250\rceil = 1250,$$
$$n_\text{hop} = \lceil 5\times10^{-3}\times50\,000 \rceil = \lceil 250\rceil = 250,$$
$$T_\text{SVD}^\star = 25 + 2\times5 = 35\,\text{ms}, \quad T_\text{SVD} = (1250+2\times250)/50\,000\times10^3 = 35\,\text{ms}.$$

---

### 3.4 Modo `by_revolution`, `Ai_length_mode = "total_window"` — derivación completa

Se desea que la ventana SVD cubra un span total de $K$ unidades temporales (revoluciones o periodos modales). La misma fórmula de inversión de la Sección 2.4 se aplica aquí, con $n = A_i$:

$$K(A_i) = N_\text{win} + (A_i - 1)\cdot\text{step} \quad [\text{unidades}].$$

Invirtiendo:

$$A_i - 1 = \frac{K - N_\text{win}}{\text{step}} \implies A_i = \frac{K - N_\text{win}}{\text{step}} + 1.$$

El menor entero que garantiza $K(A_i) \geq K$:

$$\boxed{A_i = \left\lceil \frac{K - N_\text{win}}{\text{step}} + 1 \right\rceil}$$

Span real resultante:

$$K_\text{real} = N_\text{win} + (A_i - 1)\cdot\text{step} \geq K, \quad K_\text{real} - K \in [0, \text{step}).$$

Restricción de validez: $K > N_\text{win}$ (mismo argumento que en Sección 2.4).

**Ejemplo numérico** ($N_\text{win} = 5$, $\text{step} = 5$, $K_\text{rev\_svd} = 15$):

$$A_i = \left\lceil \frac{15-5}{5} + 1 \right\rceil = \lceil 3 \rceil = 3. \quad K_\text{real} = 5 + 2\times5 = 15. \quad\text{(exacto)}$$

**Ejemplo con exceso** ($N_\text{win} = 5$, $\text{step} = 5$, $K_\text{rev\_svd} = 14$):

$$A_i = \left\lceil \frac{14-5}{5}+1 \right\rceil = \lceil 2.8 \rceil = 3. \quad K_\text{real} = 15 > 14.$$

---

### 3.5 Modo `by_modal`, `Ai_length_mode = "frames"` — derivación completa

Idéntico a la Sección 3.3 con $T_\text{unit} = T_\text{modal}$. Todos los pasos y fórmulas se replican sustituyendo $T_\text{rev} \to T_\text{modal}$, `N_rev_window` $\to$ `N_modal_window` y `step_rev` $\to$ `step_modal`.

**Ejemplo numérico** ($f_s = 50\,000$ Hz, $f_\text{modal} = 150$ Hz, $N_\text{win} = 5$, $\text{step} = 1$, $A_i = 3$):

$$T_\text{modal} = 1/150\,\text{s}, \quad w = 5\times(1000/150)\,\text{ms} = 33.3\overline{3}\,\text{ms},$$
$$h = 1\times(1000/150)\,\text{ms} = 6.6\overline{6}\,\text{ms},$$
$$n_\text{win} = \lceil 33.3\overline{3}\times10^{-3}\times50\,000\rceil = \lceil1666.\overline{6}\rceil = 1667,$$
$$n_\text{hop} = \lceil 6.6\overline{6}\times10^{-3}\times50\,000\rceil = \lceil333.\overline{3}\rceil = 334,$$
$$w_\text{ef} = 1667/50\,000\times10^3 = 33.34\,\text{ms},\quad h_\text{ef} = 334/50\,000\times10^3 = 6.68\,\text{ms}.$$

---

## 4. Resultados matemáticos transversales

### 4.1 Fórmula de inversión de la ventana deslizante (demostración general)

Sea una secuencia de $n$ ventanas, cada una de ancho $N$ unidades, con avance $s$ unidades entre inicios consecutivos ($1 \leq s \leq N$). El inicio de la ventana $i$ ($i = 1, \ldots, n$) es

$$p_i = (i-1)\cdot s,$$

y el fin de la ventana $i$ es $p_i + N = (i-1)\cdot s + N$. El **span total** (desde el inicio de la ventana 1 hasta el fin de la ventana $n$) es

$$K(n) = (n-1)\cdot s + N.$$

Dado un span deseado $K^\star > N$, se busca el menor $n \in \mathbb{Z}^+$ tal que $K(n) \geq K^\star$:

$$K(n) \geq K^\star \iff (n-1)\cdot s \geq K^\star - N \iff n \geq \frac{K^\star - N}{s} + 1.$$

El menor entero que satisface esta condición es

$$\boxed{n^\star = \left\lceil \frac{K^\star - N}{s} + 1 \right\rceil.}$$

El span real con $n^\star$ ventanas es $K(n^\star) = (n^\star-1)\cdot s + N$. El exceso sobre el objetivo es

$$K(n^\star) - K^\star = \left(\left\lceil \frac{K^\star-N}{s} \right\rceil - \frac{K^\star-N}{s}\right)\cdot s \in [0, s).$$

La fórmula es la misma para los tres indicadores y para todos los modos, independientemente de la unidad física de $K^\star$, $N$ y $s$ (revoluciones, periodos modales, frames RMS, columnas STFT). Solo cambian los nombres de las variables.

**Casos particulares:**
- $K^\star = N + (m-1)\cdot s$ para algún $m \in \mathbb{Z}^+$: el argumento del `ceil` es entero exacto y $n^\star = m$ (sin exceso).
- $K^\star = N + 1$ con $s = 1$: $n^\star = 2$ (mínimo con solapamiento).

---

### 4.2 Acotación del error de cuantización por función `ceil`

**Proposición.** Sea $x = N_\text{phys} \cdot T \cdot f_s \geq 0$ con $N_\text{phys}, T, f_s > 0$. Entonces la duración real del bloque de $\lceil x \rceil$ muestras satisface

$$t_\text{real} = \frac{\lceil x \rceil}{f_s} \in \left[t_\text{target},\; t_\text{target} + \frac{1}{f_s}\right),$$

donde $t_\text{target} = N_\text{phys} \cdot T$. El error absoluto $\Delta t = t_\text{real} - t_\text{target} \in [0, 1/f_s)$.

**Demostración.** Por definición de `ceil`, $\lceil x \rceil = x + \varepsilon$ con $\varepsilon \in [0,1)$. Entonces

$$t_\text{real} = \frac{\lceil x \rceil}{f_s} = \frac{x+\varepsilon}{f_s} = \frac{x}{f_s} + \frac{\varepsilon}{f_s} = t_\text{target} + \frac{\varepsilon}{f_s}.$$

Como $\varepsilon \in [0,1)$, se tiene $\Delta t = \varepsilon/f_s \in [0, 1/f_s)$.

**Corolario.** El error relativo es

$$\frac{\Delta t}{t_\text{target}} = \frac{\varepsilon}{x} < \frac{1}{x} = \frac{1}{N_\text{phys} \cdot T \cdot f_s}.$$

Para los parámetros típicos del caso de estudio ($f_s = 50\,000$ Hz, $T \geq T_\text{rev} = 5$ ms, $N_\text{phys} \geq 1$):

$$\frac{\Delta t}{t_\text{target}} < \frac{1}{1 \times 0.005 \times 50\,000} = \frac{1}{250} = 0.4\%.$$

Para ventanas de 5 periodos o más el error relativo baja a $< 0.08\%$.

---

### 4.3 Idempotencia de la doble cuantización (`ceil` compuesto)

En el indicador SSQ-STFT, el resolver convierte primero de unidades físicas a milisegundos ($w = N_\text{win}\cdot T_\text{unit}\times10^3$), luego calcula los valores efectivos en ms ($w_\text{ef} = \lceil w\cdot10^{-3}\cdot f_s\rceil/f_s\times10^3$) y los pasa al pipeline. El pipeline aplica a su vez $n_\text{win}' = \lceil w_\text{ef}\cdot10^{-3}\cdot f_s\rceil$.

**Proposición.** $n_\text{win}' = n_\text{win}$.

**Demostración.**

$$w_\text{ef} = \frac{\lceil w\cdot10^{-3}\cdot f_s\rceil}{f_s}\times10^3 \implies w_\text{ef}\cdot10^{-3}\cdot f_s = \lceil w\cdot10^{-3}\cdot f_s\rceil \in \mathbb{Z}.$$

Luego $n_\text{win}' = \lceil w_\text{ef}\cdot10^{-3}\cdot f_s\rceil = \lceil \lceil w\cdot10^{-3}\cdot f_s\rceil \rceil = \lceil w\cdot10^{-3}\cdot f_s\rceil = n_\text{win}$.

La segunda igualdad usa que $\lceil m \rceil = m$ para cualquier $m \in \mathbb{Z}$.

---

### 4.4 Resumen de todas las fórmulas de conversión

La tabla siguiente concentra cada conversión de parámetros físicos a nativos, con la fórmula exacta y la unidad de cada lado.

| Indicador | Modo | Parámetro nativo | Fórmula | Tipo |
|---|---|---|---|---|
| MaxEnt | `by_revolution` | $N_\text{seg}$ | $N_\text{seg} = N_\text{rev}$ | Exacta (entero) |
| MaxEnt | `by_revolution` | $n_\text{rpm}$ | $n_\text{rpm} = 60/T_\text{rev}$ | Exacta |
| MaxEnt | `by_revolution` | $\text{step\_seg}$ | $\text{step\_seg} = \text{step\_rev}$ | Exacta (entero) |
| MaxEnt | `by_modal` | $N_\text{seg}$ | $N_\text{seg} = N_m$ | Exacta (entero modal) |
| MaxEnt | `by_modal` | $n_\text{rpm,modal}$ | $n_\text{rpm,modal} = 60/T_\text{modal}$ | Exacta (re-escala) |
| MaxEnt | `by_modal` | $\text{step\_seg}$ | $\text{step\_seg} = \text{step\_modal}$ | Exacta (entero) |
| MaxEnt | `raw`+`by_revolution` | $N_\text{samples}$ | $\lceil N_\text{rev}\cdot T_\text{rev}\cdot f_s\rceil$ | `ceil` |
| MaxEnt | `raw`+`by_revolution` | $\text{step\_samples}$ | $\lceil\text{step\_rev}\cdot T_\text{rev}\cdot f_s\rceil$ | `ceil` |
| MaxEnt | `raw`+`by_modal` | $N_\text{samples}$ | $\lceil N_m\cdot T_\text{modal}\cdot f_s\rceil$ | `ceil` |
| MaxEnt | `raw`+`by_modal` | $\text{step\_samples}$ | $\lceil\text{step\_modal}\cdot T_\text{modal}\cdot f_s\rceil$ | `ceil` |
| RMS-CV | `by_*` | $W$ | $\lceil N_\text{win}\cdot T_\text{unit}\cdot f_s\rceil$ | `ceil` |
| RMS-CV | `by_*` | $\rho$ | $1 - \text{step}/N_\text{win}$ | Exacta |
| RMS-CV | `by_*/frames` | $m$ | $m = m_\text{input}$ | Directa |
| RMS-CV | `by_*/total` | $m$ | $\lceil(K-N_\text{win})/\text{step}+1\rceil$ | `ceil`-inversión |
| SSQ | `by_*` | $w$ [ms] | $N_\text{win}\cdot T_\text{unit}\times10^3$ | Exacta (float) |
| SSQ | `by_*` | $h$ [ms] | $\text{step}\cdot T_\text{unit}\times10^3$ | Exacta (float) |
| SSQ | `by_*/frames` | $A_i$ | $A_i = A_{i,\text{input}}$ | Directa |
| SSQ | `by_*/total` | $A_i$ | $\lceil(K-N_\text{win})/\text{step}+1\rceil$ | `ceil`-inversión |

En todos los modos, el resolver también calcula las cantidades de trazabilidad: duración exacta vs. real de la ventana ($t_\text{win}^\star$ vs. $t_\text{win}$), paso exacto vs. real ($\Delta t^\star$ vs. $\Delta t_\text{RMS}$), span total exacto vs. real ($T_\text{CV}^\star$ vs. $T_\text{CV}$ o $T_\text{SVD}^\star$ vs. $T_\text{SVD}$), todos expresados tanto en segundos como en unidades físicas ($T_\text{unit}$).
