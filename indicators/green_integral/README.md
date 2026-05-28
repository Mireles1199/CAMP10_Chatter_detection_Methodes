# green_integral

Detector de chatter basado en el **método de la integral de Green** (área en el espacio de fases).

## Principio

Para cada ventana temporal se calculan las áreas de los ciclos en el plano desplazamiento–velocidad
mediante la integral de línea de Green:

$$A_n = \frac{1}{2}\left|\oint (x\,dv - v\,dx)\right|$$

El indicador $\delta_n$ mide la tasa de crecimiento log-relativa entre ciclos consecutivos:

$$\delta_n = -\text{LOG\_CTC} \cdot \text{median}\bigl(\ln A_{n+1} - \ln A_n\bigr)$$

Un valor $\delta_n < 0$ indica crecimiento de amplitud → **inestabilidad / chatter**.

## Instalación

```bash
cd indicators/green_integral
pip install -e .
```

## Uso rápido

```python
from green_integral import SignalData, GreenIntegralConfig, run_green_integral, plots_green_integral
from green_integral.logging_setup import configure_logging, LOGGING_LEVELS

configure_logging(level=LOGGING_LEVELS["info"])

sig = SignalData(
    t=t, displacement=x, velocity=v, fs=fs, name="mi_senal"
)
config = GreenIntegralConfig(
    f_modal=150.0,
    num_T=6,
    dt=1e-2,
    data_filtrated=True,
)
result = run_green_integral(sig, config)
plots_green_integral(signal=sig, result=result)
```

## Niveles de logging

| Nivel      | Contenido                                    |
|------------|----------------------------------------------|
| `warning`  | solo resultado crítico                        |
| `info`     | resultado + configuración del indicador       |
| `info_plus`| + progreso por ventana (cada 100)            |
| `debug`    | + detalle interno por ventana (áreas, r_n…)  |
