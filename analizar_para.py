import os
import logging
import warnings
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict

# Suppress warnings for clean console output
warnings.filterwarnings('ignore')

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and Configuration
EXPERIMENT_FILE = "experiment_results.csv"
BASELINE_FILE = "baseline_results.csv"
OUTPUT_EXCEL = "cuadros_tesis_mipa_completo.xlsx"

PROFILE_MAP = {
    'user_vulnerable': 'Perfil A (Vulnerable)',
    'user_curioso':    'Perfil B (Curioso)',
    'user_control':    'Perfil C (Control)'
}

INTENT_MAP = {
    "SOPORTE_EMOCIONAL":    "ESC-01 Desahogo / Protector",
    "CONSULTA_INFORMATIVA": "ESC-02 Consulta / Sabio",
    "DIALOGO_SOCIAL":       "ESC-03 Social / Bufón",
    "CRISIS_ALERTA":        "ESC-04 Crisis / PROTOCOL_SAFETY"
}

SCENARIO_ORDER = [
    "ESC-01 Desahogo / Protector",
    "ESC-02 Consulta / Sabio",
    "ESC-03 Social / Bufón",
    "ESC-04 Crisis / PROTOCOL_SAFETY"
]

PROFILE_ORDER = ['Perfil A (Vulnerable)', 'Perfil B (Curioso)', 'Perfil C (Control)']

def calculate_adjusted_coherence(row: pd.Series) -> int:
    """
    Calculates adjusted coherence, strictly crediting ethical interventions 
    triggered by the Jester guardrail.

    Args:
        row (pd.Series): A single row from the experiment results DataFrame.

    Returns:
        int: 1 if coherent or ethically adjusted, 0 otherwise.
    """
    if row.get('coherence', row.get('coherencia')) == 1:
        return 1
    
    if row.get('mges_reason') == 'ethics_guardrail_jester':
        return 1
        
    return 0

def bootstrap_ci(data: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """
    Calculates the confidence interval for the mean using bootstrapping.

    Args:
        data (np.ndarray): The 1D array of numerical data.
        confidence (float): The desired confidence level.
        n_bootstrap (int): Number of bootstrap resamples.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[float, float]: The lower and upper bounds of the confidence interval.
    """
    rng = np.random.default_rng(seed)
    if len(data) == 0:
        return (np.nan, np.nan)
        
    n = len(data)
    means = [np.mean(rng.choice(data, size=n, replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)
    
    return (lower, upper)

def calc_chi2_test(df: pd.DataFrame, col_coherence: str = 'adjusted_coherence') -> Tuple[float, float, int, np.ndarray]:
    """
    Performs a Chi-squared test of independence between user profiles and coherence.

    Args:
        df (pd.DataFrame): The main results DataFrame.
        col_coherence (str): The column name containing the coherence metric.

    Returns:
        Tuple: Chi2 statistic, p-value, degrees of freedom, and expected frequencies.
    """
    contingency_table = pd.crosstab(df['Perfil'], df[col_coherence])
    
    for val in [0, 1]:
        if val not in contingency_table.columns:
            contingency_table[val] = 0
            
    contingency_table = contingency_table[[0, 1]]
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    return chi2, p, dof, expected

def describe_series(s: pd.Series) -> Dict[str, float]:
    """Generates standard descriptive statistics for a pandas Series."""
    return {
        'Media (ms)': np.mean(s),
        'Mediana (ms)': np.median(s),
        'Desv. Est. (ms)': np.std(s, ddof=1),
        'Mínimo (ms)': np.min(s),
        'Máximo (ms)': np.max(s),
        'P95 (ms)': np.percentile(s, 95)
    }

def main() -> None:
    logger.info("Loading experiment data...")
    if not os.path.exists(EXPERIMENT_FILE):
        raise FileNotFoundError(f"Missing {EXPERIMENT_FILE}. Run run_experiment.py first.")
        
    df = pd.read_csv(EXPERIMENT_FILE)

    if df['mges_intervened'].dtype == object:
        df['mges_intervened'] = df['mges_intervened'].astype(str).str.strip().str.lower().map({'true': True, 'false': False})
    df['mges_intervened'] = df['mges_intervened'].astype(bool)

    df['Perfil'] = df['user_id'].map(PROFILE_MAP)
    df['Escenario'] = df['expected_intent'].map(INTENT_MAP)
    
    df['adjusted_coherence'] = df.apply(calculate_adjusted_coherence, axis=1)

    logger.info("Calculating Table 9 (Adjusted Coherence with CI)...")
    means_adj = df.pivot_table(
        index='Escenario', columns='Perfil', values='adjusted_coherence', aggfunc='mean'
    ) * 100
    means_adj = means_adj.reindex(SCENARIO_ORDER)
    means_adj['Media'] = means_adj.mean(axis=1)

    ci_lower = pd.DataFrame(index=SCENARIO_ORDER, columns=PROFILE_ORDER)
    ci_upper = pd.DataFrame(index=SCENARIO_ORDER, columns=PROFILE_ORDER)

    for scenario in SCENARIO_ORDER:
        for profile in PROFILE_ORDER:
            subset = df[(df['Escenario'] == scenario) & (df['Perfil'] == profile)]['adjusted_coherence']
            if len(subset) > 0:
                low, up = bootstrap_ci(subset)
                ci_lower.loc[scenario, profile] = low * 100
                ci_upper.loc[scenario, profile] = up * 100
            else:
                ci_lower.loc[scenario, profile] = np.nan
                ci_upper.loc[scenario, profile] = np.nan

    table_9 = pd.DataFrame(index=SCENARIO_ORDER, columns=PROFILE_ORDER + ['Media'])
    for scenario in SCENARIO_ORDER:
        for profile in PROFILE_ORDER:
            mean_val = means_adj.loc[scenario, profile]
            low = ci_lower.loc[scenario, profile]
            up = ci_upper.loc[scenario, profile]
            if not np.isnan(low) and not np.isnan(up):
                table_9.loc[scenario, profile] = f"{mean_val:.1f}% [{low:.1f}–{up:.1f}]"
            else:
                table_9.loc[scenario, profile] = f"{mean_val:.1f}%"
        table_9.loc[scenario, 'Media'] = f"{means_adj.loc[scenario, 'Media']:.1f}%"

    logger.info("Calculating Table 9B (Raw Coherence)...")
    # Handling both English/Spanish naming from CSV variations
    coherence_col = 'coherence' if 'coherence' in df.columns else 'coherencia'
    means_raw = df.pivot_table(
        index='Escenario', columns='Perfil', values=coherence_col, aggfunc='mean'
    ) * 100
    means_raw = means_raw.reindex(SCENARIO_ORDER)
    means_raw['Media'] = means_raw.mean(axis=1)

    table_9_raw = pd.DataFrame(index=SCENARIO_ORDER, columns=PROFILE_ORDER + ['Media'])
    for scenario in SCENARIO_ORDER:
        for profile in PROFILE_ORDER:
            mean_val = means_raw.loc[scenario, profile]
            table_9_raw.loc[scenario, profile] = f"{mean_val:.1f}%"
        table_9_raw.loc[scenario, 'Media'] = f"{means_raw.loc[scenario, 'Media']:.1f}%"

    logger.info("Performing Chi-squared tests...")
    chi2, p, dof, _ = calc_chi2_test(df, 'adjusted_coherence')
    chi2_table = pd.DataFrame({
        'Métrica': ['Chi-cuadrado', 'Valor p', 'Grados de libertad'],
        'Valor': [f"{chi2:.2f}", f"{p:.4f}", f"{dof}"]
    })

    logger.info("Calculating Table 10 (Latency metrics)...")
    latency_col = 'latency_ms' if 'latency_ms' in df.columns else 'latencia_ms'
    latency_base = df[df['archetype_change'] == 'NO'][latency_col]
    latency_switch = df[df['archetype_change'] == 'YES'][latency_col]
    
    # Fallback to 'SI' if CSV still has Spanish output
    if latency_switch.empty and 'SI' in df['archetype_change'].values:
        latency_switch = df[df['archetype_change'] == 'SI'][latency_col]

    stats_base = describe_series(latency_base)
    stats_switch = describe_series(latency_switch)

    table_10 = pd.DataFrame({
        'Métrica': list(stats_base.keys()),
        'Sin cambio de arquetipo': [f"{stats_base[k]:.1f}" for k in stats_base],
        'Con cambio de arquetipo': [f"{stats_switch[k]:.1f}" for k in stats_switch]
    })

    logger.info("Calculating Table 11 (MGES metrics)...")
    total_interactions = len(df)
    total_interventions = df['mges_intervened'].sum()
    global_rate = (total_interventions / total_interactions) * 100

    real_crisis = df[df['expected_intent'] == 'CRISIS_ALERTA']
    num_crisis = len(real_crisis)

    df['pred_crisis'] = df['archetype_detected'] == 'PROTOCOL_SAFETY'

    tp = len(df[(df['expected_intent'] == 'CRISIS_ALERTA') & df['pred_crisis']])
    fn = len(df[(df['expected_intent'] == 'CRISIS_ALERTA') & ~df['pred_crisis']])
    fp = len(df[(df['expected_intent'] != 'CRISIS_ALERTA') & df['pred_crisis']])
    tn = len(df[(df['expected_intent'] != 'CRISIS_ALERTA') & ~df['pred_crisis']])

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    intervention_by_profile = df.groupby('Perfil')['mges_intervened'].mean() * 100

    confusion_matrix = pd.DataFrame({
        'Real \ Predicho': ['Crisis (Real)', 'No Crisis (Real)'],
        'Predicho Crisis': [tp, fp],
        'Predicho No Crisis': [fn, tn]
    })

    table_11_general = pd.DataFrame({
        'Métrica': [
            'Total interacciones',
            'Intervenciones MGES totales',
            'Tasa global intervención (Rsafe)',
            'Crisis reales en test set',
            'Verdaderos positivos (TP)',
            'Falsos negativos (FN)',
            'Falsos positivos (FP)',
            'Verdaderos negativos (TN)',
            'Precisión',
            'Recall (Tasa detección crisis)',
            'F1-Score'
        ],
        'Valor': [
            total_interactions,
            total_interventions,
            f"{global_rate:.2f}%",
            num_crisis,
            tp, fn, fp, tn,
            f"{precision:.1f}%",
            f"{recall:.2f}%",
            f"{f1:.1f}%"
        ]
    })

    table_11_profile = pd.DataFrame({
        'Perfil': [p for p in PROFILE_ORDER if p in intervention_by_profile.index],
        'Tasa intervención MGES (%)': [
            f"{intervention_by_profile[p]:.2f}%" for p in PROFILE_ORDER if p in intervention_by_profile.index
        ]
    })

    comparison_df = None
    mcnemar_df = None
    if os.path.exists(BASELINE_FILE):
        logger.info("Comparing with baseline results...")
        df_base = pd.read_csv(BASELINE_FILE)
        df_base['Perfil'] = df_base['user_id'].map(PROFILE_MAP)
        df_base['Escenario'] = df_base['expected_intent'].map(INTENT_MAP)
        
        base_coherence_col = 'raw_coherence' if 'raw_coherence' in df_base.columns else 'coherencia_cruda'
        baseline_coherence = df_base.groupby(['Escenario', 'Perfil'])[base_coherence_col].mean() * 100
        baseline_coherence = baseline_coherence.unstack(fill_value=0)
        baseline_coherence = baseline_coherence.reindex(index=SCENARIO_ORDER, columns=PROFILE_ORDER)
        
        comp = means_adj.copy()
        comparison_df = pd.DataFrame(index=SCENARIO_ORDER)
        for profile in PROFILE_ORDER:
            comparison_df[f'{profile} (Sistema)'] = comp[profile].apply(lambda x: f"{x:.1f}%")
            comparison_df[f'{profile} (Baseline)'] = baseline_coherence[profile].apply(lambda x: f"{x:.1f}%")
        
        from statsmodels.stats.contingency_tables import mcnemar
        table_2x2 = np.array([
            [((df['adjusted_coherence'] == 1) & (df_base[base_coherence_col] == 1)).sum(),
             ((df['adjusted_coherence'] == 1) & (df_base[base_coherence_col] == 0)).sum()],
            [((df['adjusted_coherence'] == 0) & (df_base[base_coherence_col] == 1)).sum(),
             ((df['adjusted_coherence'] == 0) & (df_base[base_coherence_col] == 0)).sum()]
        ])
        result = mcnemar(table_2x2, exact=True)
        mcnemar_df = pd.DataFrame({
            'Prueba': ['McNemar (exacto)'],
            'p-valor': [result.pvalue]
        })
    else:
        logger.warning("Baseline file not found. Skipping McNemar comparison.")

    logger.info("Exporting results to %s...", OUTPUT_EXCEL)
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        table_9.to_excel(writer, sheet_name='Tabla9_Coherencia_IC')
        table_9_raw.to_excel(writer, sheet_name='Tabla9_Coherencia_Cruda') 
        chi2_table.to_excel(writer, sheet_name='Chi2_Perfiles', index=False)
        table_10.to_excel(writer, sheet_name='Tabla10_Latencia', index=False)
        table_11_general.to_excel(writer, sheet_name='Tabla11_MGES_General', index=False)
        table_11_profile.to_excel(writer, sheet_name='Tabla11_MGES_Perfil', index=False)
        confusion_matrix.to_excel(writer, sheet_name='Matriz_Confusion', index=False)
        
        if comparison_df is not None:
            comparison_df.to_excel(writer, sheet_name='Comparacion_Baseline')
        if mcnemar_df is not None:
            mcnemar_df.to_excel(writer, sheet_name='McNemar', index=False)

    logger.info("Analysis complete! Summary:")
    logger.info(" - Global adjusted coherence: %.1f%%", df['adjusted_coherence'].mean() * 100)
    logger.info(" - Global raw coherence: %.1f%%", df[coherence_col].mean() * 100)
    logger.info(" - MGES intervention rate: %.2f%%", global_rate)
    logger.info(" - Crisis Recall: %.2f%%", recall)
    
    if p < 0.05:
        logger.info(" - Profile differences: SIGNIFICANT (p = %.4f)", p)
    else:
        logger.info(" - Profile differences: NOT significant (p = %.4f)", p)

if __name__ == "__main__":
    main()