"""
Dashboard unificado de series temporales SHF
-------------------------------------------

Este script combina lo mejor de dos implementaciones previas de dashboards:
`improved_dashboard_Version2.py` y `prueba_estadisticos6 (2).py`.  La
principal mejora es la robustez en la carga de datos (manejo de delimitadores
de CSV, codificaciones y formatos de fecha), la detección inteligente de
columnas de fecha, valor y categoría, y la generación de visualizaciones
interactivas con descomposición de series temporales que selecciona
automáticamente el modo aditivo o multiplicativo minimizando el error.

Características destacadas
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Carga de datos flexible**: admite CSV, Excel y Parquet.  Para archivos
  CSV se intenta varias codificaciones y se detecta el delimitador.
* **Detección de columnas**: sugiere automáticamente columnas de fecha,
  valor (numéricas) y categoría, con la opción de especificar un formato de
  fecha personalizado.
* **Análisis robusto de fechas**: reconoce cuando una columna contiene
  años (e.g. "2000") y los convierte a fechas («2000-01-01»).  También
  intenta detectar el formato de fecha cuando el análisis automático falla.
* **Preprocesamiento con caché**: convierte columnas de valor a numérico,
  crea una columna de categoría predeterminada si no se especifica una,
  calcula tasas de crecimiento y etiquetas de periodo según la frecuencia
  seleccionada.
* **Opciones avanzadas**: permite filtrar por rango de fechas y detectar
  valores atípicos mediante z-score (opcional).
* **Descomposición automática**: utiliza la librería `darts` para extraer
  tendencia y estacionalidad, eligiendo el modo (aditivo/multiplicativo)
  según el error de reconstrucción.  El resultado se muestra en un
  dashboard interactivo con Plotly.

Para ejecutar este archivo en un entorno local, utiliza el comando:

```
streamlit run dashboard_unificado.py
```

Se requiere tener instaladas las dependencias: `streamlit`, `pandas`,
`numpy`, `plotly`, `darts` y `openpyxl` (para Excel).
"""

import calendar
import re
from datetime import datetime
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from darts import TimeSeries
from darts.utils.statistics import ModelMode, extract_trend_and_seasonality
from plotly.subplots import make_subplots
import streamlit as st


# -----------------------------------------------------------------------------
# Configuración inicial y supresión de advertencias
# -----------------------------------------------------------------------------
# Configurar la página de Streamlit
st.set_page_config(page_title="📊 Dashboard SHF", layout="wide", page_icon="📊")

# Suprimir advertencias específicas de Darts que pueden ser ruidosas
warnings.filterwarnings("ignore", message=".*frequency of the time series is*")
warnings.filterwarnings("ignore", message=".*Series has NaN values.*")


# -----------------------------------------------------------------------------
# Funciones utilitarias de procesamiento de datos y detección de columnas
# -----------------------------------------------------------------------------

@st.cache_data
def convert_df(df: pd.DataFrame) -> bytes:
    """Convierte un DataFrame a CSV en bytes y utiliza caché para evitar
    recomputaciones costosas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a convertir.

    Returns
    -------
    bytes
        Representación en CSV codificada en UTF-8.
    """
    return df.to_csv(index=False).encode("utf-8")


def detect_date_format(sample_dates: List[str]) -> Optional[str]:
    """Intenta detectar el formato de fecha predominante en una muestra.

    Parameters
    ----------
    sample_dates : list of str
        Lista de cadenas de fecha de ejemplo.

    Returns
    -------
    str or None
        Formato de fecha detectado (por ejemplo, "%Y-%m-%d"), o None si no
        se detecta un formato dominante.
    """
    if not sample_dates:
        return None
    # Formatos de fecha comunes para probar
    formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%m-%d-%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
        "%d.%m.%Y", "%Y.%m.%d", "%m.%d.%Y",
        "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
    ]
    # Convertir a cadena y eliminar vacíos
    sample_dates = [str(d) for d in sample_dates if pd.notna(d) and str(d).strip()]
    for fmt in formats:
        success = 0
        for date_str in sample_dates:
            try:
                datetime.strptime(date_str, fmt)
                success += 1
            except Exception:
                pass
        if success / len(sample_dates) > 0.5:
            return fmt
    return None


def _parse_numeric_years(series: pd.Series) -> Optional[pd.Series]:
    """Detecta si una serie contiene años y los convierte a fechas.

    Una serie es considerada de años si todos los valores no nulos son
    enteros o flotantes y se encuentran en el rango [1000, 3000], o si son
    cadenas de cuatro dígitos.  En ese caso se devuelve una serie de
    fechas en el 1 de enero de cada año.  Si no cumple con el criterio,
    devuelve None.
    """
    non_null = series.dropna()
    if non_null.empty:
        return None
    year_like = False
    try:
        if pd.api.types.is_integer_dtype(non_null) or pd.api.types.is_float_dtype(non_null):
            vals = non_null.astype(int)
            year_like = ((vals >= 1000) & (vals <= 3000)).all()
        else:
            str_vals = non_null.astype(str).str.strip()
            year_like = str_vals.str.fullmatch(r"\d{4}").all()
    except Exception:
        year_like = False
    if year_like:
        return pd.to_datetime(series.astype(str) + "-01-01", format="%Y-%m-%d", errors="coerce")
    return None


def parse_dates(
    df: pd.DataFrame, col_name: str, custom_format: Optional[str] = None
) -> pd.Series:
    """Analiza una columna de fechas de forma robusta.

    Primero intenta convertir la columna utilizando un formato personalizado
    especificado por el usuario.  Si esto falla, intenta detectar si la
    columna representa años y los convierte.  En caso contrario, intenta
    convertir automáticamente con pandas.  Si más del 50 % de los valores
    resultan NaT, se emplea una detección de formato sobre una muestra.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que contiene la columna de fechas.
    col_name : str
        Nombre de la columna de fechas seleccionada por el usuario.
    custom_format : str, optional
        Formato de fecha especificado por el usuario (por ejemplo, "%d/%m/%Y").

    Returns
    -------
    pd.Series
        Serie de fechas analizada.  Puede contener NaT en caso de valores
        no interpretables.
    """
    series = df[col_name]
    if custom_format:
        try:
            return pd.to_datetime(series, format=custom_format, errors="coerce")
        except Exception as e:
            st.warning(f"No se pudo analizar la columna '{col_name}' con el formato especificado: {e}")
    # Intentar detectar años numéricos
    year_parsed = _parse_numeric_years(series)
    if year_parsed is not None:
        return year_parsed
    # Conversión automática
    dates = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    if dates.isna().sum() > 0.5 * len(dates):
        # Demasiados NaT: intentar detectar formato
        sample = series.dropna().astype(str).head(20).tolist()
        detected_format = detect_date_format(sample)
        if detected_format:
            parsed = pd.to_datetime(series, format=detected_format, errors="coerce")
            st.info(f"Formato de fecha detectado: {detected_format}")
            return parsed
    return dates


def guess_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Devuelve una lista de columnas que probablemente contienen valores numéricos."""
    numerics = df.select_dtypes(include=["number"]).columns.tolist()
    # Verificar si columnas de texto pueden convertirse a números
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(10)
        if not sample.empty:
            try:
                sample.astype(float)
                numerics.append(col)
            except Exception:
                pass
    return numerics


def guess_date_columns(df: pd.DataFrame) -> List[str]:
    """Devuelve una lista de columnas que probablemente contienen fechas."""
    date_cols: List[str] = df.select_dtypes(include=["datetime"]).columns.tolist()
    # Verificar otras columnas de tipo objeto con patrones de fecha
    date_pattern = r"\d{1,4}[-/\. ]\d{1,2}[-/\. ]\d{1,4}|\d{1,2}\s+[a-zA-Z]{3,9}\s+\d{2,4}"
    for col in df.select_dtypes(include=["object", "int64", "float64"]).columns:
        if col in date_cols:
            continue
        sample = df[col].dropna().astype(str).head(10)
        if not sample.empty:
            matches = [bool(re.search(date_pattern, str(x))) for x in sample]
            # Si la mayoría de la muestra coincide con el patrón, considerarla fecha
            if sample.dtype != "datetime64[ns]" and sum(matches) / len(matches) > 0.5:
                date_cols.append(col)
    return date_cols


def guess_category_columns(
    df: pd.DataFrame, min_categories: int = 2, max_categories: int = 30
) -> List[str]:
    """Devuelve una lista de columnas que probablemente sean categóricas."""
    cols: List[str] = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if (df[col].dtype == "object" and nunique <= max_categories) or (
            min_categories <= nunique <= max_categories
        ):
            cols.append(col)
    return cols


def add_period_labels(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Añade una columna 'Periodo' con etiquetas legibles según la frecuencia."""
    if freq == "Q":
        quarters = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
        df["Periodo"] = df["Fecha"].dt.quarter.map(quarters)
    elif freq == "M":
        months_dict = {i: calendar.month_abbr[i] for i in range(1, 13)}
        df["Periodo"] = df["Fecha"].dt.month.map(months_dict)
    elif freq == "W":
        df["Periodo"] = "Semana " + df["Fecha"].dt.isocalendar().week.astype(str)
    elif freq == "A":
        df["Periodo"] = df["Fecha"].dt.year.astype(str)
    else:
        dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dias_es = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        mapping = dict(zip(dias, dias_es))
        df["Periodo"] = df["Fecha"].dt.day_name().map(mapping)
    return df


@st.cache_data
def preprocess_data(
    df: pd.DataFrame,
    col_fecha: str,
    col_valor: str,
    col_cat: Optional[str],
    freq: str,
    custom_date_format: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Preprocesa los datos seleccionados por el usuario.

    Convierte fechas, valida valores numéricos, crea una categoría por defecto
    cuando no se especifica una, ordena los datos, añade etiquetas de
    periodo y calcula tasas de crecimiento.
    """
    df_proc = df.copy()
    # Convertir fechas
    df_proc["Fecha"] = parse_dates(df_proc, col_fecha, custom_date_format)
    df_proc = df_proc.dropna(subset=["Fecha"])
    if df_proc.empty:
        st.error("No se pudo interpretar ninguna fecha. Comprueba el formato de la columna seleccionada.")
        return None
    # Convertir valores numéricos
    if df_proc[col_valor].dtype not in ["int64", "float64"]:
        try:
            df_proc[col_valor] = pd.to_numeric(df_proc[col_valor], errors="coerce")
            st.info(f"Columna '{col_valor}' convertida a valores numéricos")
        except Exception:
            st.error(f"La columna '{col_valor}' debe contener valores numéricos")
            return None
    # Manejo de categorías
    if col_cat is None or col_cat == col_fecha:
        df_proc["__cat__"] = "Serie única"
        col_cat = "__cat__"
    elif df_proc[col_cat].nunique() <= 1 or df_proc[col_cat].nunique() > len(df_proc) / 2:
        df_proc["__cat__"] = "Serie única"
        col_cat = "__cat__"
        st.info("No se encontraron categorías significativas; se usará una única serie")
    # Ordenar datos
    df_proc = df_proc.sort_values([col_cat, "Fecha"]).reset_index(drop=True)
    # Añadir etiquetas de periodo
    df_proc = add_period_labels(df_proc, freq)
    # Calcular crecimiento
    df_proc["log_ind"] = np.log(df_proc[col_valor].replace(0, np.nan))
    df_proc["log_lag"] = df_proc.groupby(col_cat, observed=True)["log_ind"].diff()
    df_proc["g"] = (np.exp(df_proc["log_lag"]) - 1) * 100
    return df_proc


def elegir_modelo(ts: TimeSeries) -> ModelMode:
    """Elige el mejor modo (aditivo o multiplicativo) para la descomposición."""
    # Si hay valores no positivos, usar aditivo
    if (ts.values() <= 0).any():
        return ModelMode.ADDITIVE
    errores: Dict[ModelMode, float] = {}
    for modo in (ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE):
        try:
            trend_ts, season_ts = extract_trend_and_seasonality(ts=ts, freq=None, model=modo, method="naive")
            if modo == ModelMode.ADDITIVE:
                reconstruido = trend_ts + season_ts
            else:
                reconstruido = trend_ts * season_ts
            err = float(np.mean((ts.values().ravel() - reconstruido.values().ravel()) ** 2))
            errores[modo] = err
        except Exception:
            errores[modo] = float("inf")
    return min(errores, key=errores.get)


def _ts_to_series(tseries: TimeSeries) -> pd.Series:
    """Convierte un objeto TimeSeries de Darts a pandas Series de forma robusta."""
    for method in ("pd_series", "to_series", "to_pandas", "pd_dataframe"):
        if hasattr(tseries, method):
            res = getattr(tseries, method)()
            if isinstance(res, pd.DataFrame) and res.shape[1] >= 1:
                return res.iloc[:, 0]
            if isinstance(res, pd.Series):
                return res
    vals = tseries.values()
    arr = vals[:, 0, 0] if getattr(vals, "ndim", 0) == 3 else vals.ravel()
    idx = getattr(tseries, "time_index", pd.RangeIndex(len(arr)))
    return pd.Series(arr, index=idx)


@st.cache_data
def compute_decomposition(df_cat: pd.DataFrame, col_valor: str) -> Optional[Dict[str, pd.Series]]:
    """Calcula la descomposición de tendencia y estacionalidad para una categoría."""
    try:
        ts = TimeSeries.from_dataframe(df_cat, "Fecha", col_valor, fill_missing_dates=True)
        model_mode = elegir_modelo(ts)
        trend_ts, season_ts = extract_trend_and_seasonality(ts=ts, freq=None, model=model_mode, method="naive")
        trend_s = _ts_to_series(trend_ts).reindex(df_cat["Fecha"]).rename("Tendencia")
        season_s = _ts_to_series(season_ts).reindex(df_cat["Fecha"]).rename("Estacionalidad")
        return {
            "trend": trend_s,
            "season": season_s,
            "model_mode": model_mode,
        }
    except Exception as e:
        st.error(f"Error en la descomposición: {str(e)}")
        return None


def has_outliers(series: pd.Series, threshold: float = 3.0) -> bool:
    """Detecta valores atípicos en una serie usando z-score."""
    if len(series) < 4:
        return False
    z_scores = np.abs((series - series.mean()) / series.std(ddof=0))
    return (z_scores > threshold).any()


def main() -> None:
    """Función principal que construye la interfaz y ejecuta el flujo del dashboard."""
    # Cabecera de la aplicación
    st.title("📊 Dashboard SHF Interactivo")
    # Añadir logo corporativo de Realytics para una identidad visual coherente
    try:
        st.image("realytics.jpg", width=150)
    except Exception:
        pass
    st.markdown(
        """
        Este dashboard permite analizar datos temporales con descomposición de series.
        Carga un archivo, selecciona las columnas relevantes y explora los componentes de la serie.
        """
    )
    # Contenedor para carga de archivos
    upload_container = st.container()
    with upload_container:
        uploaded_file = st.file_uploader(
            "Sube tu archivo (.csv, .xlsx, .parquet)", type=["csv", "xlsx", "parquet"], key="file_uploader"
        )
        if uploaded_file:
            # Manejo de carga de CSV con detección de delimitador y codificación
            try:
                if uploaded_file.name.endswith(".csv"):
                    try:
                        df = pd.read_csv(uploaded_file)
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding="latin1")
                    except Exception:
                        # Detectar delimitador leyendo una muestra de 1KB
                        uploaded_file.seek(0)
                        sample = uploaded_file.read(1024).decode("utf-8", errors="ignore")
                        uploaded_file.seek(0)
                        if "," in sample:
                            df = pd.read_csv(uploaded_file, sep=",")
                        elif ";" in sample:
                            df = pd.read_csv(uploaded_file, sep=";")
                        elif "\t" in sample:
                            df = pd.read_csv(uploaded_file, sep="\t")
                        else:
                            st.error("No se pudo determinar el delimitador del archivo CSV")
                            return
                elif uploaded_file.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith(".parquet"):
                    df = pd.read_parquet(uploaded_file)
                else:
                    st.error("Formato de archivo no soportado")
                    return
            except Exception as e:
                st.error(f"Error al cargar el archivo: {e}")
                return
            # Mostrar métricas básicas y vista previa dentro de un expander para una interfaz más limpia
            with st.expander("📄 Detalles del dataset", expanded=False):
                n_rows, n_cols = df.shape
                n_missing = int(df.isna().sum().sum())
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Filas", f"{n_rows:,}")
                with col2:
                    st.metric("Columnas", n_cols)
                with col3:
                    total = n_rows * n_cols
                    missing_pct = n_missing / total if total > 0 else 0
                    st.metric("Datos faltantes", f"{n_missing:,} ({missing_pct:.1%})")
                st.dataframe(df.head())
                # Botón de descarga para datos crudos
                csv_bytes = convert_df(df)
                st.download_button(
                    "Descargar datos cargados (CSV)",
                    csv_bytes,
                    "datos_cargados.csv",
                    "text/csv",
                    key="download-raw-csv",
                )
            # Selección de columnas con disposición horizontal para una interfaz más compacta
            st.subheader("Selección de columnas")
            date_cols = guess_date_columns(df)
            numeric_cols = guess_numeric_columns(df)
            category_cols = guess_category_columns(df)
            sel_col1, sel_col2, sel_col3, sel_col4 = st.columns(4)
            with sel_col1:
                col_fecha = st.selectbox(
                    "Fecha", options=date_cols + [c for c in df.columns if c not in date_cols], key="col_fecha"
                )
            with sel_col2:
                col_valor = st.selectbox(
                    "Valor", options=numeric_cols + [c for c in df.columns if c not in numeric_cols], key="col_valor"
                )
            with sel_col3:
                col_cat = st.selectbox(
                    "Categoría", options=["[No usar categorías]"] + category_cols + [c for c in df.columns if c not in category_cols], key="col_cat"
                )
                if col_cat == "[No usar categorías]":
                    col_cat = None
            with sel_col4:
                # Selección de frecuencia
                mapa_freq = {
                    "Diaria": "D",
                    "Semanal": "W",
                    "Mensual": "M",
                    "Trimestral": "Q",
                    "Anual": "A",
                }
                freq_es = st.selectbox(
                    "Frecuencia", list(mapa_freq.keys()), index=2, key="freq"
                )
                freq_final = mapa_freq[freq_es]
            # Formato de fecha personalizado y opciones avanzadas en un expander
            with st.expander("Opciones", expanded=False):
                use_custom_format = st.checkbox(
                    "Usar formato de fecha personalizado", value=False, key="use_custom_format"
                )
                custom_date_format = None
                if use_custom_format:
                    custom_date_format = st.text_input(
                        "Formato de fecha (ej: %d/%m/%Y)",
                        help=(
                            "Ingresa el formato de fecha según la convención de Python. Ejemplos:\n"
                            "%d/%m/%Y para 31/12/2023\n%Y-%m-%d para 2023-12-31"
                        ),
                        key="custom_date_format",
                    )
                handle_outliers = st.checkbox(
                    "Detectar valores atípicos", value=False, key="outliers"
                )
                date_range_filter = st.checkbox(
                    "Filtrar por rango de fechas", value=False, key="date_range_filter"
                )
                date_range = None
                if date_range_filter:
                    temp_dates = parse_dates(df, col_fecha, custom_date_format)
                    if temp_dates.notna().any():
                        min_date = temp_dates.min().date()
                        max_date = temp_dates.max().date()
                        date_range = st.date_input(
                            "Selecciona el rango de fechas",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            key="date_range",
                        )
            # Procesamiento de datos
            with st.spinner("Procesando datos..."):
                processed_df = preprocess_data(df, col_fecha, col_valor, col_cat, freq_final, custom_date_format)
                if processed_df is None:
                    return
                # Aplicar filtro de fechas
                if date_range_filter and date_range and len(date_range) == 2:
                    start_date, end_date = date_range
                    processed_df = processed_df[(processed_df["Fecha"].dt.date >= start_date) & (processed_df["Fecha"].dt.date <= end_date)]
                    st.info(f"Datos filtrados entre {start_date} y {end_date}")
                # Verificar si la columna de categoría tiene un único valor
                actual_cat = col_cat or "__cat__"
                if processed_df[actual_cat].nunique() == 1:
                    st.warning(
                        f"La columna de categoría seleccionada solo contiene un valor único. Se utilizará 'Serie única' como categoría."
                    )
                    processed_df["__cat__"] = "Serie única"
                    actual_cat = "__cat__"
                # Lista de categorías
                categorias = processed_df[actual_cat].unique().tolist()
                if not categorias:
                    st.error("No se encontraron categorías válidas para el análisis")
                    return
                # Mostrar estadísticas básicas en un expander para una interfaz más limpia
                st.subheader("Estadísticas básicas")
                with st.expander("Ver estadísticas", expanded=False):
                    col_stats1, col_stats2 = st.columns(2)
                    with col_stats1:
                        st.write("Resumen de valores")
                        st.dataframe(processed_df[col_valor].describe())
                    with col_stats2:
                        st.write("Conteo por categoría")
                        cat_counts = (
                            processed_df.groupby(actual_cat)
                            .size()
                            .reset_index(name="count")
                        )
                        st.dataframe(cat_counts)
                # Advertencias de calidad de datos
                warnings_list: List[str] = []
                # Fechas faltantes (solo para frecuencia diaria)
                if freq_final == "D":
                    for cat in categorias:
                        cat_data = processed_df[processed_df[actual_cat] == cat]
                        date_range_full = pd.date_range(cat_data["Fecha"].min(), cat_data["Fecha"].max(), freq="D")
                        missing_dates = len(date_range_full) - len(cat_data)
                        if missing_dates > 0 and missing_dates / len(date_range_full) > 0.1:
                            warnings_list.append(
                                f"⚠️ Categoría '{cat}': {missing_dates} fechas faltantes ({missing_dates / len(date_range_full):.1%})"
                            )
                # Valores atípicos
                if handle_outliers:
                    for cat in categorias:
                        cat_data = processed_df[processed_df[actual_cat] == cat]
                        if has_outliers(cat_data[col_valor]):
                            warnings_list.append(f"⚠️ Categoría '{cat}': se detectaron valores atípicos")
                if warnings_list:
                    with st.expander("⚠️ Advertencias de calidad de datos"):
                        for w in warnings_list:
                            st.warning(w)
                # Descomposición por categoría
                st.subheader("Descomposición y visualización")
                per_cat: Dict[str, Dict[str, pd.Series]] = {}
                progress_bar = st.progress(0)
                for idx, cat in enumerate(categorias):
                    sub_df = processed_df[processed_df[actual_cat] == cat]
                    if len(sub_df) < 4:
                        st.warning(f"Categoría '{cat}' tiene muy pocos datos ({len(sub_df)}) y se omitirá")
                        continue
                    decomp = compute_decomposition(sub_df, col_valor)
                    if decomp:
                        per_cat[cat] = {
                            "x": sub_df["Fecha"],
                            "y": sub_df[col_valor],
                            "y_g": sub_df["g"],
                            "x_box": sub_df["Periodo"],
                            "y_box": sub_df["g"],
                            "y_trend": decomp["trend"],
                            "y_season": decomp["season"],
                            "model": "Multiplicativo" if decomp["model_mode"] == ModelMode.MULTIPLICATIVE else "Aditivo",
                        }
                        # Marcar outliers si corresponde
                        if handle_outliers and has_outliers(sub_df[col_valor]):
                            z_scores = np.abs((sub_df[col_valor] - sub_df[col_valor].mean()) / sub_df[col_valor].std(ddof=0))
                            outliers = sub_df[z_scores > 3]
                            per_cat[cat]["outliers_x"] = outliers["Fecha"]
                            per_cat[cat]["outliers_y"] = outliers[col_valor]
                    progress_bar.progress((idx + 1) / len(categorias))
                if not per_cat:
                    st.error("No se pudieron generar visualizaciones. Verifica la calidad de los datos.")
                    return
                # Mostrar tipo de modelo para la primera categoría
                first_cat = next(iter(per_cat))
                st.info(f"Modelo detectado para la primera categoría ('{first_cat}'): {per_cat[first_cat]['model']}")
                # Construir figura con distribución de filas personalizada para evitar superposición
                fig = make_subplots(
                    rows=5,
                    cols=1,
                    row_heights=[3, 2, 2, 2, 2],
                    vertical_spacing=0.04,
                    subplot_titles=(
                        "Serie original",
                        "Crecimiento (%)",
                        "Estacionalidad",
                        "Tendencia",
                        "Distribución estacional (boxplot)",
                    ),
                )
                trace_map: Dict[str, List[int]] = {}
                trace_idx = 0
                for cat in categorias:
                    if cat not in per_cat:
                        continue
                    s = per_cat[cat]
                    # Serie original
                    fig.add_trace(
                        go.Scatter(
                            x=s["x"], y=s["y"], mode="lines+markers", name=f"Serie - {cat}", visible=False, line=dict(width=2)
                        ),
                        row=1,
                        col=1,
                    )
                    idx1 = trace_idx
                    trace_idx += 1
                    # Outliers en serie original
                    if "outliers_x" in s:
                        fig.add_trace(
                            go.Scatter(
                                x=s["outliers_x"],
                                y=s["outliers_y"],
                                mode="markers",
                                marker=dict(color="red", size=10, symbol="x"),
                                name=f"Outliers - {cat}",
                                visible=False,
                            ),
                            row=1,
                            col=1,
                        )
                        trace_idx += 1
                    # Crecimiento
                    fig.add_trace(
                        go.Scatter(
                            x=s["x"], y=s["y_g"], mode="lines+markers", name=f"Crec. - {cat}", visible=False, line=dict(width=2)
                        ),
                        row=2,
                        col=1,
                    )
                    idx2 = trace_idx
                    trace_idx += 1
                    # Estacionalidad
                    fig.add_trace(
                        go.Scatter(
                            x=s["x"],
                            y=s["y_season"],
                            mode="lines+markers",
                            name=f"Estacionalidad - {cat}",
                            visible=False,
                            marker=dict(symbol="square"),
                            line=dict(width=2),
                        ),
                        row=3,
                        col=1,
                    )
                    idx3 = trace_idx
                    trace_idx += 1
                    # Tendencia
                    fig.add_trace(
                        go.Scatter(
                            x=s["x"],
                            y=s["y_trend"],
                            mode="lines",
                            name=f"Tendencia - {cat}",
                            visible=False,
                            line=dict(width=2, dash="dash"),
                        ),
                        row=4,
                        col=1,
                    )
                    idx4 = trace_idx
                    trace_idx += 1
                    # Boxplot
                    fig.add_trace(
                        go.Box(
                            x=s["x_box"],
                            y=s["y_box"],
                            boxmean=True,
                            name=f"Box estacionalidad - {cat}",
                            visible=False,
                        ),
                        row=5,
                        col=1,
                    )
                    idx5 = trace_idx
                    trace_idx += 1
                    trace_map[cat] = [idx1, idx2, idx3, idx4, idx5]
                # Crear botones
                n_traces = len(fig.data)
                buttons = []
                for cat in categorias:
                    if cat not in trace_map:
                        continue
                    vis = [False] * n_traces
                    for k in trace_map[cat]:
                        vis[k] = True
                    buttons.append(
                        dict(
                            label=str(cat),
                            method="update",
                            args=[{"visible": vis}, {"title": f"Dashboard SHF - {cat}"}],
                        )
                    )
                # Configuración inicial de visibilidad para la primera categoría
                if categorias:
                    inicio = categorias[0]
                    if inicio in trace_map:
                        initial_vis = [False] * n_traces
                        for idx in trace_map[inicio]:
                            initial_vis[idx] = True
                        for i, tr in enumerate(fig.data):
                            tr.visible = initial_vis[i]
                    fig.update_layout(
                        updatemenus=[dict(buttons=buttons, direction="down", x=0.01, y=1.15)],
                        title=f"Dashboard SHF - {inicio}",
                        height=1600,
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    )
                    # Ocultar etiquetas del eje x en las primeras cuatro filas para evitar superposición
                    for row_num in range(1, 5):
                        fig.update_xaxes(showticklabels=False, row=row_num, col=1)
                # Ordenar la x del boxplot según periodo
                orden = []
                if freq_final == "Q":
                    orden = ["Q1", "Q2", "Q3", "Q4"]
                elif freq_final == "M":
                    orden = [calendar.month_abbr[i] for i in range(1, 13)]
                elif freq_final == "W":
                    orden = sorted(processed_df["Periodo"].unique().tolist(), key=lambda x: int(x.split(" ")[1]))
                elif freq_final == "A":
                    orden = sorted(processed_df["Periodo"].unique().tolist())
                else:
                    orden = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
                fig.update_xaxes(categoryorder="array", categoryarray=orden, row=5, col=1)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Carga un archivo para comenzar el análisis.")


if __name__ == "__main__":
    main()