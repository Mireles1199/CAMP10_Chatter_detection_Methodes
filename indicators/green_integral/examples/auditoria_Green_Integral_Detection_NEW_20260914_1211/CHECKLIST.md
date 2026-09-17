# Checklist accionable — `Green_Integral_Detection_NEW.py`

Cada ítem referencia la sección correspondiente de `AUDITORIA.md`.

## Prioridad CRÍTICA (rompe funcionalidad documentada)

- [ ] Arreglar `run_green_std` para `func="Default"`: asignar `t_d_no_FAR`
      también en esa rama (o dar un valor por defecto `None` antes del
      `if/else`), y hacer que el bloque de logging (`t_d[0]`, `.t_d.size`,
      `.t_wins`) no asuma la forma de `FixedWindowResult` cuando
      `raw_result` es un `GreenIntegralResult`.
      — `runner_std.py:240-262, 277, 301-313, 327` — AUDITORIA.md §3.1.1

## Prioridad ALTA (puede romper con datos/configs razonables)

- [ ] Proteger `filter_window_signals(q_win, q_o_win)` en
      `_process_one_window` con la misma guarda `len(q_win) >= 7` que ya
      existe en `runner_fixed.py`, para evitar `ValueError` de
      `savgol_filter` en ventanas cortas.
      — `window_processor.py:45-46` — AUDITORIA.md §3.1.2
- [ ] Corregir (o documentar de verdad) el uso de `config.t_theorical` en
      el cálculo de `t_d_no_FAR`: o bien dar un valor por defecto seguro
      cuando es `None` (p. ej. `-inf`, o desactivar el filtro de FAR), o
      bien actualizar el docstring que dice "no afecta la detección".
      — `runner_fixed.py:1009` / `utils/types.py:240` — AUDITORIA.md §3.1.3
- [ ] Corregir `Contour_Line_Area.window_duration` para que refleje los
      límites reales de la ventana cuando hay menos de 2 cruces completos
      (actualmente queda en `[0.0, 0.0]`), o documentar explícitamente que
      ese caso depende de la guarda externa en `window_processor.py`.
      — `contour_area.py:59, 184-189` — AUDITORIA.md §3.1.4
- [ ] Corregir la doble multiplicación por `dt` en
      `_estimate_f0_autocorr` (`T0 = lags[mask][idx_peak] * dt` →
      `T0 = lags[mask][idx_peak]`).
      — `zero_crossing.py:181` — AUDITORIA.md §3.1.5

## Prioridad MEDIA (riesgos / fragilidad)

- [ ] Sustituir la llamada a `scipy.optimize.minimize` (Nelder–Mead) por
      el cálculo directo de la media (`np.mean`), que es la solución
      analítica exacta del mismo problema — elimina una sobrecarga de
      cómputo repetida por cada ciclo de cada ventana.
      — `contour_area.py:216-225` — AUDITORIA.md §3.2.1
- [ ] Añadir `plt.close(fig)` a las figuras de debug de
      `runner_fixed.py` que no llegan a `plt.show()`, y evaluar si el
      mapa de winding number (bucle 500×500) puede vectorizarse o
      reducirse de resolución.
      — `runner_fixed.py:478-866` — AUDITORIA.md §3.2.2
- [ ] Añadir un piso de ruido (`area_noise_eps`-like) también en la
      variante `Default`, antes de tomar `np.log(A)` en
      `compute_delta_n`.
      — `delta_n.py:87-96` — AUDITORIA.md §3.2.3
- [ ] Acotar el `except Exception` de `process_windows_serial` a los
      tipos de error realmente esperados, o al menos distinguir en el log
      entre "ventana con datos insuficientes" (esperado) y "error de
      programación" (no esperado).
      — `window_processor.py:158-302` — AUDITORIA.md §3.2.4
- [ ] Validar (o al menos advertir) si la señal no está uniformemente
      muestreada, en vez de asumir `dt = t[1]-t[0]` para toda la señal.
      — `Green_Integral_Detection_NEW.py:213`, `runner_fixed.py:425`,
      `green_integral_plots.py:651` — AUDITORIA.md §3.2.5
- [ ] Sustituir `self.points_cicles_t[i+1, 1]` / `[i, 1]` (usado para
      forzar velocidad=0 en los extremos del ciclo) por un literal `0.0`
      explícito o una constante nombrada, para no depender de una
      coincidencia estructural de otro array.
      — `contour_area.py:201, 205` — AUDITORIA.md §3.2.6
- [ ] Decidir una única función "dueña" de configurar
      `matplotlib.rcParams` al importar `green_integral.viz` (eliminar la
      duplicidad entre `plots.py` y `green_integral_plots.py`), o mover
      esa configuración a una llamada explícita en vez de un efecto
      secundario del import.
      — `plots.py:28-49`, `green_integral_plots.py:30-62` —
      AUDITORIA.md §3.2.7

## Prioridad BAJA (limpieza / legibilidad / consistencia)

- [ ] Eliminar los imports sin usar `statistics` y
      `matplotlib.pyplot as plt`.
      — `window_processor.py:6, 10` — AUDITORIA.md §3.3.1
- [ ] Corregir la etiqueta duplicada "Total Windows" (imprime
      `N_cycles_per_seg` en vez del número total de ventanas analizadas).
      — `runner_std.py:288-289` — AUDITORIA.md §3.3.2
- [ ] Cambiar `params_physical["use_area_threshold"]` por
      `params_physical.get("use_area_threshold", False)` para
      consistencia con el resto de la función y evitar un `KeyError`
      evitable.
      — `runner_std.py:294` — AUDITORIA.md §3.3.3
- [ ] Unificar el signo de `K_out` entre los modos `"none"`/`"mean"` y
      `"median"` de `cycle_area_norm`.
      — `runner_fixed.py:868-884` — AUDITORIA.md §3.3.4
- [ ] Eliminar el bloque de código comentado/duplicado al inicio del
      bucle principal de `_fixed_window_pipeline`.
      — `runner_fixed.py:454-464` — AUDITORIA.md §3.3.5
- [ ] Unificar el tipo de `t_d`/`t_d_no_FAR` entre `GreenIntegralResult`
      y `FixedWindowResult` (causa raíz del hallazgo crítico §3.1.1).
      — `utils/types.py:132, 275-276` — AUDITORIA.md §3.3.6
- [ ] Vectorizar `_integrate_G_sliding` usando la integral acumulada ya
      calculada en `_integrate_G` más `np.searchsorted`, en vez del doble
      bucle Python actual.
      — `runner_fixed.py:249-282` — AUDITORIA.md §3.3.7
- [ ] Documentar o externalizar las rutas HDF5 absolutas hardcodeadas del
      script de ejemplo.
      — `Green_Integral_Detection_NEW.py:74-96` — AUDITORIA.md §3.3.8
