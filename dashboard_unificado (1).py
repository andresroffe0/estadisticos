"""Se requiere tener instaladas las dependencias: `streamlit`, `pandas`,
`numpy`, `plotly`, `darts` y `openpyxl` (para Excel).
"""

import calendar
from datetime import datetime
import warnings
from typing import Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.statistics import ModelMode, extract_trend_and_seasonality
import streamlit as st

# Ocultar la barra de Streamlit (hamburger menu y footer)
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Configuración inicial y supresión de advertencias
# -----------------------------------------------------------------------------
# Configurar la página de Streamlit
st.set_page_config(page_title="Línea de Tiempo", layout="wide", page_icon="📊")

# Configuración global de Altair para evitar límites de filas y simplificar la estética
alt.data_transformers.disable_max_rows()
alt.renderers.set_embed_options(actions=False)

# Suprimir advertencias específicas de Darts que pueden ser ruidosas
warnings.filterwarnings("ignore", message=".*frequency of the time series is*")
warnings.filterwarnings("ignore", message=".*Series has NaN values.*")


# -----------------------------------------------------------------------------
# Funciones utilitarias de procesamiento de datos y detección de columnas
# -----------------------------------------------------------------------------

GROWTH_OFFSETS_BY_FREQ = {
    "A": {"YoY": pd.DateOffset(years=1)},
    "Q": {"QoQ": pd.DateOffset(months=3), "YoY": pd.DateOffset(years=1)},
    "M": {"MoM": pd.DateOffset(months=1), "YoY": pd.DateOffset(years=1)},
    "W": {"WoW": pd.DateOffset(weeks=1), "YoY": pd.DateOffset(years=1)},
    "D": {
        "DoD": pd.DateOffset(days=1),
        "WoW": pd.DateOffset(weeks=1),
        "MoM": pd.DateOffset(months=1),
        "YoY": pd.DateOffset(years=1),
    },
}

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


def calculate_growth(
    df: pd.DataFrame,
    cat_col: str,
    value_col: str,
    freq: str,
    growth_type: str,
) -> pd.DataFrame:
    """Calcula la variación porcentual alineada a periodos calendario.

    Parameters
    ----------
    df : pd.DataFrame
        Datos procesados que incluyen las columnas de categoría y fecha.
    cat_col : str
        Nombre de la columna que identifica la categoría.
    value_col : str
        Nombre de la columna con los valores numéricos.
    freq : str
        Frecuencia seleccionada ("A", "Q", "M", "W" o "D").
    growth_type : str
        Métrica de crecimiento a calcular (por ejemplo, "YoY", "QoQ").

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas `Fecha`, la categoría y la variación porcentual
        etiquetada como `Crecimiento`.
    """

    offsets = GROWTH_OFFSETS_BY_FREQ.get(freq, {})
    offset = offsets.get(growth_type)
    if offset is None:
        return pd.DataFrame(columns=["Fecha", cat_col, "Crecimiento"])

    growth_frames: List[pd.DataFrame] = []
    for categoria, grupo in df.groupby(cat_col):
        if grupo.empty:
            continue
        serie = grupo.sort_values("Fecha").set_index("Fecha")[value_col]
        comparativo = serie.shift(freq=offset)
        crecimiento = (serie / comparativo - 1) * 100
        temp = crecimiento.rename("Crecimiento").reset_index()
        temp[cat_col] = categoria
        growth_frames.append(temp)

    if not growth_frames:
        return pd.DataFrame(columns=["Fecha", cat_col, "Crecimiento"])

    return pd.concat(growth_frames, ignore_index=True)


def main() -> None:
    """Función principal que construye la interfaz y ejecuta el flujo del dashboard."""
    # Cabecera de la aplicación
    st.title("Línea de Tiempo")
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
            all_columns = df.columns.tolist()
            sel_col1, sel_col2, sel_col3, sel_col4 = st.columns(4)
            with sel_col1:
                col_fecha = st.selectbox("Fecha", options=all_columns, key="col_fecha")
            remaining_for_value = [c for c in all_columns if c != col_fecha]
            if not remaining_for_value:
                st.error("Selecciona un archivo que contenga al menos otra columna para el valor numérico.")
                return
            with sel_col2:
                col_valor = st.selectbox("Valor", options=remaining_for_value, key="col_valor")
            remaining_for_category = [c for c in all_columns if c not in {col_fecha, col_valor}]
            with sel_col3:
                cat_option = st.selectbox(
                    "Categoría",
                    options=["[No usar categorías]"] + remaining_for_category,
                    key="col_cat",
                )
                col_cat = None if cat_option == "[No usar categorías]" else cat_option
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
            show_custom_format = st.checkbox(
                "Ingresar formato de fecha personalizado",
                value=False,
                key="toggle_custom_date_format",
            )
            custom_date_format = None
            if show_custom_format:
                custom_date_format = (
                    st.text_input(
                        "Formato de fecha (ej. %d/%m/%Y)",
                        value="",
                        key="custom_date_format",
                    ).strip()
                    or None
                )
            handle_outliers = False
            date_range_filter = False
            date_range = None
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
                    processed_df["__cat__"] = "Serie única"
                    actual_cat = "__cat__"
                # Lista de categorías
                categorias = processed_df[actual_cat].unique().tolist()
                if not categorias:
                    st.error("No se encontraron categorías válidas para el análisis")
                    return
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
                # Calcular métricas comparativas para visualizaciones Altair
                processed_df["Promedio pares"] = (
                    processed_df.groupby("Fecha")[col_valor].transform("mean")
                )
                processed_df["Delta vs pares"] = processed_df[col_valor] - processed_df["Promedio pares"]

                # Descomposición por categoría
                st.info("Analizando componentes por categoría…")
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

                palette = {
                    "serie": "#006d77",
                    "pares": "#83c5be",
                    "delta_pos": "#55a630",
                    "delta_neg": "#d62828",
                    "growth": "#023e8a",
                    "trend": "#9d4edd",
                    "season": "#ffb703",
                }

                seleccionadas = st.multiselect(
                    "Selecciona categorías para visualizar",
                    categorias,
                    default=[categorias[0]] if categorias else [],
                )

                if not seleccionadas:
                    st.warning("Selecciona al menos una categoría para mostrar las tarjetas analíticas.")

                metric_labels = {
                    col_valor: "Serie",
                    "Promedio pares": "Promedio de pares",
                    "Delta vs pares": "Delta vs pares",
                }

                valid_categories = [cat for cat in seleccionadas if cat in per_cat]
                missing_categories = set(seleccionadas) - set(valid_categories)
                if missing_categories:
                    st.warning(
                        "Las siguientes categorías no pudieron procesarse: "
                        + ", ".join(sorted(missing_categories))
                    )

                if valid_categories:
                    seleccion_df = processed_df[processed_df[actual_cat].isin(valid_categories)].copy()

                    with st.container(border=True):
                        st.markdown("**Crecimiento porcentual**")
                        opciones_crecimiento = GROWTH_OFFSETS_BY_FREQ.get(freq_final, {})
                        if not opciones_crecimiento:
                            st.info("No hay métricas de crecimiento disponibles para la frecuencia seleccionada.")
                        else:
                            growth_type = st.selectbox(
                                "Tipo de crecimiento",
                                list(opciones_crecimiento.keys()),
                                key="growth_type_chart",
                            )
                            growth_df = calculate_growth(
                                seleccion_df,
                                actual_cat,
                                col_valor,
                                freq_final,
                                growth_type,
                            )
                            growth_df = growth_df.dropna(subset=["Crecimiento"])
                            has_growth_rows = len(growth_df.index) > 0
                            if not has_growth_rows:
                                st.info("No hay datos suficientes para calcular el crecimiento seleccionado.")
                            else:
                                growth_chart = (
                                    alt.Chart(growth_df.rename(columns={actual_cat: "Categoría"}))
                                    .mark_line(point=True)
                                    .encode(
                                        x=alt.X("Fecha:T", title="Fecha"),
                                        y=alt.Y("Crecimiento:Q", title="Crecimiento (%)"),
                                        color=alt.Color("Categoría:N", title="Categoría"),
                                    )
                                    .properties(height=240)
                                    .configure_axis(grid=False)
                                    .configure_legend(orient="bottom")
                                )
                                st.altair_chart(growth_chart, use_container_width=True)

                    trend_frames = [
                        pd.DataFrame(
                            {
                                "Fecha": per_cat[cat]["x"],
                                "Tendencia": per_cat[cat]["y_trend"].values,
                                "Categoría": cat,
                            }
                        )
                        for cat in valid_categories
                    ]
                    combined_trend = (
                        pd.concat(trend_frames, ignore_index=True)
                        if trend_frames
                        else pd.DataFrame(columns=["Fecha", "Tendencia", "Categoría"])
                    )
                    combined_trend = combined_trend.dropna(subset=["Tendencia"])
                    has_trend_rows = len(combined_trend.index) > 0
                    if has_trend_rows:
                        tendencia_chart = (
                            alt.Chart(combined_trend)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("Fecha:T", title="Fecha"),
                                y=alt.Y("Tendencia:Q", title="Tendencia"),
                                color=alt.Color("Categoría:N", title="Categoría"),
                            )
                            .properties(height=240)
                            .configure_axis(grid=False)
                            .configure_legend(orient="bottom")
                        )
                        with st.container(border=True):
                            st.markdown("**Tendencia por categoría**")
                            st.altair_chart(tendencia_chart, use_container_width=True)

                for cat in valid_categories:
                    st.markdown(f"### {cat}")
                    cat_df = processed_df[processed_df[actual_cat] == cat].copy()
                    if cat_df.empty:
                        st.warning(f"No hay datos disponibles para la categoría '{cat}'.")
                        continue

                    cat_df["Signo delta"] = np.where(cat_df["Delta vs pares"] >= 0, "Por encima", "Por debajo")

                    melted = cat_df.melt(
                        id_vars=["Fecha", actual_cat],
                        value_vars=list(metric_labels.keys()),
                        var_name="variable",
                        value_name="Valor",
                    )
                    melted["Métrica"] = melted["variable"].map(metric_labels)

                    line_df = melted[melted["Métrica"].isin(["Serie", "Promedio de pares"])].copy()
                    linea_chart = (
                        alt.Chart(line_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("Fecha:T", title="Fecha"),
                            y=alt.Y("Valor:Q", title="Nivel"),
                            color=alt.Color(
                                "Métrica:N",
                                scale=alt.Scale(range=[palette["serie"], palette["pares"]]),
                                legend=alt.Legend(orient="bottom"),
                            ),
                        )
                        .properties(height=260)
                        .configure_axis(grid=False)
                        .configure_legend(orient="bottom")
                    )

                    delta_chart = (
                        alt.Chart(cat_df)
                        .mark_area(opacity=0.6)
                        .encode(
                            x=alt.X("Fecha:T", title="Fecha"),
                            y=alt.Y("Delta vs pares:Q", title="Diferencia"),
                            color=alt.Color(
                                "Signo delta:N",
                                scale=alt.Scale(range=[palette["delta_pos"], palette["delta_neg"]]),
                                legend=alt.Legend(title="", orient="bottom"),
                            ),
                        )
                        .properties(height=260)
                        .configure_axis(grid=False)
                        .configure_legend(orient="bottom")
                    )

                    col_linea, col_delta = st.columns(2)
                    with col_linea:
                        with st.container(border=True):
                            st.markdown("**Serie vs. promedio de pares**")
                            st.altair_chart(linea_chart, use_container_width=True)
                    with col_delta:
                        with st.container(border=True):
                            st.markdown("**Delta contra pares**")
                            st.altair_chart(delta_chart, use_container_width=True)

                    season_series = per_cat[cat]["y_season"].copy()
                    season_title = "Estacionalidad"
                    if per_cat[cat]["model"] == "Multiplicativo":
                        season_series = (season_series - 1) * 100
                        season_title = "Estacionalidad (%)"

                    tendencia_estacionalidad = pd.DataFrame(
                        {
                            "Fecha": per_cat[cat]["x"],
                            "Estacionalidad": season_series.values,
                        }
                    )

                    estacionalidad_chart = (
                        alt.Chart(tendencia_estacionalidad)
                        .mark_line(color=palette["season"])
                        .encode(
                            x=alt.X("Fecha:T", title="Fecha"),
                            y=alt.Y("Estacionalidad:Q", title=season_title),
                        )
                        .properties(height=220)
                        .configure_axis(grid=False)
                    )

                    with st.container(border=True):
                        st.markdown("**Estacionalidad**")
                        st.altair_chart(estacionalidad_chart, use_container_width=True)

                st.subheader("Estadísticas básicas")
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
        else:
            st.info("Carga un archivo para comenzar el análisis.")


if __name__ == "__main__":
    main()
