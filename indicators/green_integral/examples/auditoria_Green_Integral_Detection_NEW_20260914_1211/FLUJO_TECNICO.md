# Flujo técnico — `Green_Integral_Detection_NEW.py`

Grafo de dependencias real (imports/llamadas), en orden de ejecución.
Los nodos marcados con **[INACTIVO]** solo se alcanzan si se cambia la
configuración por defecto del script (`USE_FIXED_WINDOW=True`,
`hilbert=False`, `debug_level<2`, `t_theorical` provisto).

```mermaid
flowchart TD
    ENTRY["examples/Green_Integral_Detection_NEW.py"]

    subgraph PKG["green_integral (paquete)"]
        INIT["__init__.py<br/>registra INFO_PLUS_LEVEL"]
        LOGSETUP["logging_setup.py<br/>configure_logging()"]
        TYPES["utils/types.py<br/>SignalData, StdSignalData,<br/>GreenIntegralConfig/Result,<br/>FixedWindowConfig/Result,<br/>IndicatorResult"]
        HDF5["utils/hdf5_utils.py<br/>HDF5Reader"]
    end

    ENTRY -->|"1. configure_logging()"| LOGSETUP
    LOGSETUP --> INIT
    ENTRY -->|"2. import API pública"| INIT
    INIT --> TYPES
    INIT --> HDF5
    INIT -->|"import"| RUNSTD["lib/runner_std.py<br/>run_green_std()"]
    INIT -->|"import"| VIZMAIN["viz/green_integral_plots.py<br/>plots_green_integral()<br/>plots_fixed_window()<br/>plots_signal_diagnostics()"]

    ENTRY -->|"3. HDF5Reader(path)"| HDF5
    ENTRY -->|"4. StdSignalData(...)"| TYPES
    ENTRY -->|"5. run_green_std(sig_std, config_used)"| RUNSTD

    RUNSTD -->|"resuelve f_cycle/N_cycles_per_seg → f_modal/num_T/dt"| RUNSTD
    RUNSTD -->|"func='Default' [INACTIVO por defecto]"| RUNDEF["lib/runner.py<br/>run_green_integral()"]
    RUNSTD -->|"func='FixedWindow' [ACTIVO]"| RUNFIX["lib/runner_fixed.py<br/>run_fixed_window()"]

    subgraph DEFAULT["Variante Default (clustering)"]
        RUNDEF -->|"DebugManager"| DBG["utils/debug.py<br/>DebugManager"]
        RUNDEF -->|"process_windows_serial()"| WPROC["lib/window_processor.py"]
        WPROC -->|"filter_window_signals()"| SFILT["utils/signal_filter.py<br/>savgol_filter_window"]
        WPROC -->|"hilbert=False → Simple_ZeroCrossing"| ZCSIMPLE["utils/zero_crossing.py<br/>Simple_ZeroCrossing"]
        WPROC -->|"hilbert=True [INACTIVO]"| ZCHIL["utils/zero_crossing.py<br/>ZeroCrossing_Hilbert<br/>(bug: _estimate_f0_autocorr)"]
        WPROC -->|"Contour_Line_Area(...)"| CONTOUR["utils/contour_area.py<br/>Contour_Line_Area<br/>(minimize() redundante)"]
        CONTOUR -->|"cycles_cluster_points≠None"| CGROUP["utils/cycle_grouper.py<br/>CrossingGrouper"]
        WPROC -->|"compute_delta_n()"| DELTAN["lib/delta_n.py"]
        RUNDEF -->|"build_cycle_groups()"| CYCGRP["lib/cycle_groups.py"]
        RUNDEF -->|"μ±zσ umbral área (opcional)"| RUNDEF
    end

    subgraph FIXED["Variante FixedWindow (activa por defecto)"]
        RUNFIX -->|"savgol_filter_window()"| SFILT
        RUNFIX -->|"extract_complete_cycles()<br/>(cruces v=0 interpolados)"| RUNFIX
        RUNFIX -->|"shoelace / shoelace_oriented /<br/>open_contribution / closure_contribution"| RUNFIX
        RUNFIX -->|"_estimate_sigma()<br/>ratio | frozen_time (TheilSen/polyfit)"| RUNFIX
        RUNFIX -->|"_apply_ewma() [opcional]"| RUNFIX
        RUNFIX -->|"_integrate_G() / _integrate_G_sliding() [opcional]"| RUNFIX
        RUNFIX -->|"μ±zσ umbral área log10 (opcional)"| RUNFIX
        RUNFIX -->|"debug_level>=2 [INACTIVO]"| DIAG["lib/diagnostics.py<br/>estimate_center, center_trajectory,<br/>compute_local_phase, drift_ratio"]
    end

    RUNSTD -->|"IndicatorResult(...)<br/>[BUG: t_d_no_FAR sin asignar<br/>en rama Default]"| ENTRY

    ENTRY -->|"6. USE_FIXED_WINDOW=True → plots_fixed_window()"| VIZMAIN
    ENTRY -->|"USE_FIXED_WINDOW=False [INACTIVO]"| VIZLOCAL2["viz/green_integral_plots.py<br/>plots_green_integral()"]
    VIZLOCAL2 -->|"import"| VIZPLOTS["viz/plots.py<br/>plot_windows_local<br/>plot_windows_duration<br/>plot_indicator_local"]

    VIZMAIN -.->|"import (rcParams global,<br/>gana por orden de import)"| STYLE2["configurar_estilo_global()"]
    VIZPLOTS -.->|"import (rcParams global,<br/>sin efecto real)"| STYLE1["_configurar_estilo()"]

    style RUNSTD fill:#f66,stroke:#900,stroke-width:2px
    style RUNDEF fill:#fc9,stroke:#960
    style ZCHIL fill:#eee,stroke:#999,stroke-dasharray: 5 5
    style DIAG fill:#eee,stroke:#999,stroke-dasharray: 5 5
    style VIZLOCAL2 fill:#eee,stroke:#999,stroke-dasharray: 5 5
    style VIZPLOTS fill:#eee,stroke:#999,stroke-dasharray: 5 5
```

## Notas sobre el grafo

- **Nodo rojo** (`runner_std.py`): contiene el bug crítico §3.1.1 de
  `AUDITORIA.md` — la rama `Default` de `run_green_std` no puede
  completar su `return` sin lanzar una excepción.
- **Nodo naranja** (`runner.py`, variante Default): alcanzable hoy solo si
  se cambia `USE_FIXED_WINDOW=False` en el script de entrada; forma parte
  del flujo real (está importado y es seleccionable con una constante),
  por lo que se auditó con el mismo nivel de detalle que la ruta activa.
- **Nodos grises punteados**: código alcanzable solo bajo configuraciones
  no usadas por el script de entrada tal como está (`hilbert=True`,
  `debug_level>=2`, o la rama `USE_FIXED_WINDOW=False` de los propios
  gráficos). Se auditaron igualmente por formar parte del árbol de
  dependencias real del archivo de entrada.
- `utils/types.py` y `utils/hdf5_utils.py` son dependencias compartidas por
  prácticamente todos los módulos del paquete (se omiten flechas repetidas
  hacia ellos desde cada consumidor para no saturar el diagrama).
