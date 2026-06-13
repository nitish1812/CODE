import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.signal import find_peaks
from fpdf import FPDF
import tempfile
import os
from datetime import datetime
import itertools
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ECU Smart Dashboard", page_icon="🚗", layout="wide")

# ══════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════
for key in ['df','analyses','col_stats','pair_stats','analyses_done','ran','uploaded_file_id']:
    if key not in st.session_state:
        st.session_state[key] = None
if 'ran' not in st.session_state:
    st.session_state.ran = False

# ══════════════════════════════════════════════════════════════════
# KEYWORD MAP
# ══════════════════════════════════════════════════════════════════
KEYWORD_MAP = {
    'RPM':             ['rpm','enginespeed','engine speed','engine_speed','revs','rotations'],
    'Speed':           ['speed','velocity','vss','vehiclespeed','vehicle speed','kph','kmh'],
    'Coolant_Temp':    ['coolant','ect','coolanttemp','coolant temp','water temp','watertemp'],
    'Intake_Temp':     ['airtemp','intaketemp','iat','intake air','air temp','intakeairtemperature','airtemperature'],
    'Throttle':        ['throttle','tps','throttleposition','throttle position','throttle_pos'],
    'Engine_Load':     ['load','engineload','engine load','calculated load','engine_load'],
    'MAF':             ['maf','massair','mass air flow','airflow','mass_air'],
    'MAP':             ['map','manifold','absolutepressure','manifold pressure'],
    'STFT':            ['stft','shortfuel','short term fuel','stft1','shortfueltrim'],
    'LTFT':            ['ltft','longfuel','long term fuel','ltft1','longfueltrim'],
    'O2_Voltage':      ['o2','lambda','oxygen','lambdasensor','o2sensor','o2_voltage','o2voltage'],
    'Fuel_Level':      ['fuellevel','fuel level','fuel_level','tank'],
    'Battery_Voltage': ['battery','batteryvoltage','volt','batt','controlmodule'],
    'Ignition_Timing': ['ignition','timing','ignitiontiming','ignition timing','timingadvance'],
    'Oil_Temp':        ['oiltemp','oil temp','oil_temp','engineoil'],
    'Exhaust_Temp':    ['exhaust','egt','exhausttemp','exhaust temp'],
    'Boost_Pressure':  ['boost','boostpressure','boost pressure','turbo'],
    'Fuel_Pressure':   ['fuelpressure','fuel pressure','fuel_pressure','fuelrail'],
    'Fuel_Rate':       ['fuelrate','fuel rate','fuel_rate','consumption'],
    'Gear':            ['gear','currentgear','gear position'],
    'Accel_Pedal':     ['accelerator','pedal','accelped'],
    'Misfire':         ['misfire','misfirecount','misfire count'],
    'AFR':             ['afr','airfuelratio','air fuel ratio','targetafr'],
    'Lambda':          ['lambda','lambdasensor','wideband'],
    'Lean_Angle':      ['lean','leanangle','lean angle','tilt'],
    'BaseFuel':        ['basefuel','base fuel'],
    'BaseIgnition':    ['baseignition','base ignition'],
    'Time':            ['time','timestamp','log','datetime'],
}

# ══════════════════════════════════════════════════════════════════
# FILE PARSER
# ══════════════════════════════════════════════════════════════════
def parse_file(uploaded_file):
    content = uploaded_file.read().decode('utf-8', errors='ignore')
    lines   = content.splitlines()
    if lines and '%DataLog%' in lines[0]:
        channels, data_start = [], 0
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith('Channel :'): channels.append(s.replace('Channel : ','').strip())
            if s.startswith('Log :'):     data_start = i + 1; break
        data_lines = [l.strip() for l in lines[data_start:] if l.strip()]
        rows       = [l.split(',') for l in data_lines]
        max_len    = len(channels) + 1
        rows       = [r + ['']*(max_len-len(r)) for r in rows if len(r) > 1]
        df         = pd.DataFrame(rows, columns=['Time'] + channels)
        for col in channels: df[col] = pd.to_numeric(df[col], errors='coerce')
        return df, 'ECU Manager DataLog'
    for sep in [',',';','\t','|']:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=sep)
            if len(df.columns) > 1: return df, 'Standard CSV'
        except: continue
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file), 'Standard CSV'

def standardize(df):
    rename_map = {}
    for std_name, keywords in KEYWORD_MAP.items():
        for col in df.columns:
            col_clean = col.lower().replace(' ','').replace('_','').replace('[','').replace(']','').replace('°','').replace('/','')
            for kw in keywords:
                kw_clean = kw.lower().replace(' ','').replace('_','')
                if kw_clean in col_clean or col_clean == kw_clean:
                    if std_name not in rename_map.values():
                        rename_map[col] = std_name; break
    df = df.rename(columns=rename_map)
    for col in df.columns:
        if col != 'Time': df[col] = pd.to_numeric(df[col], errors='coerce')
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols: df[col] = df[col].fillna(df[col].median())
    if 'Time' in df.columns:
        try:
            t = pd.to_datetime('2024-01-01 ' + df['Time'].astype(str), format='%Y-%m-%d %H:%M:%S.%f')
            df['Time_Elapsed'] = (t - t.min()).dt.total_seconds()
        except:
            try:
                t = pd.to_datetime(df['Time'])
                df['Time_Elapsed'] = (t - t.min()).dt.total_seconds()
            except: df['Time_Elapsed'] = range(len(df))
    else: df['Time_Elapsed'] = range(len(df))
    if 'RPM' in df.columns:
        # Generic RPM-band labels (not vehicle-specific "highway/aggressive" claims,
        # since gearing varies a lot between vehicles)
        df['Driving_Mode'] = pd.cut(df['RPM'], bins=[0,800,2000,4000,20000],
            labels=['Idle (<800 RPM)','Low RPM (800-2000)','Mid RPM (2000-4000)','High RPM (>4000)']).astype('category')
    return df

# ══════════════════════════════════════════════════════════════════
# STATISTICAL BRAIN
# ══════════════════════════════════════════════════════════════════
def analyze_column(df, col):
    s = df[col].dropna()
    info = {
        'col': col, 'mean': float(s.mean()), 'std': float(s.std()),
        'min': float(s.min()), 'max': float(s.max()),
        'range': float(s.max()-s.min()),
        'cv': float(s.std()/(abs(s.mean())+1e-9)),
        'skewness': float(s.skew()), 'kurtosis': float(s.kurtosis()),
        'is_stable': float(s.std()) < float(abs(s.mean()))*0.05 if float(abs(s.mean())) > 0 else True,
        'has_trend': False, 'has_spikes': False, 'spike_count': 0, 'distribution': 'normal',
    }
    if 'Time_Elapsed' in df.columns and len(s) > 10:
        t = df.loc[s.index,'Time_Elapsed'].values
        slope, _, r, p, _ = stats.linregress(t, s.values)
        info.update({'trend_slope': float(slope), 'trend_r2': float(r**2),
                     'has_trend': abs(r) > 0.3 and p < 0.05,
                     'trend_dir': 'increasing' if slope > 0 else 'decreasing'})
    try:
        peaks, _ = find_peaks(s.values, height=s.mean()+2*s.std(), distance=5)
        info.update({'has_spikes': len(peaks) > 3, 'spike_count': len(peaks)})
    except: pass
    if abs(info['skewness']) > 1:   info['distribution'] = 'skewed'
    elif info['kurtosis'] > 3:      info['distribution'] = 'heavy_tailed'
    return info

def analyze_pair(df, col1, col2):
    s1, s2 = df[col1].dropna(), df[col2].dropna()
    idx    = s1.index.intersection(s2.index)
    if len(idx) < 10: return None
    s1, s2 = s1[idx], s2[idx]
    try:    r, p = stats.pearsonr(s1, s2)
    except: r, p = 0, 1
    try:    rho, _ = stats.spearmanr(s1, s2)
    except: rho    = 0
    return {
        'col1': col1, 'col2': col2,
        'pearson_r': float(r), 'pearson_p': float(p),
        'spearman_rho': float(rho),
        'is_correlated': abs(r) > 0.3 and p < 0.05,
        'is_strongly_correlated': abs(r) > 0.6,
        'is_nonlinear': abs(rho) > abs(r) + 0.15,
        'direction': 'positive' if r > 0 else 'negative',
    }

def generate_analysis_plan(df):
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
               if c not in ['Time_Elapsed'] and not c.startswith('IS_')
               and c != 'Anomaly' and df[c].std() > 0]
    if not num_cols: return [], {}, {}

    col_stats  = {col: analyze_column(df, col) for col in num_cols}
    pair_stats = {}
    for c1, c2 in itertools.combinations(num_cols, 2):
        r = analyze_pair(df, c1, c2)
        if r: pair_stats[(c1,c2)] = r

    analyses = []

    analyses.append({'id':'full_timeseries','title':'📈 All Sensor Readings Over Time','type':'timeseries_all','cols':num_cols,'score':100,'accuracy':'100%',
        'description':f'Time series of all {len(num_cols)} sensors over the session.',
        'why_useful':'Shows complete picture of every sensor at every moment. Reveals patterns, spikes and anomalies.',
        'what_found':f'{len(num_cols)} parameters recorded over {df["Time_Elapsed"].max():.0f} seconds with {len(df):,} readings.'})

    analyses.append({'id':'distributions','title':'📊 Sensor Value Distributions','type':'distributions','cols':num_cols,'score':98,'accuracy':'100%',
        'description':'Histogram of how frequently each sensor value occurred.',
        'why_useful':'Shows what values sensors spent most time at. Unusual distributions reveal operating conditions or faults.',
        'what_found':'Distribution shapes: '+', '.join([f"{c}: {col_stats[c]['distribution']}" for c in num_cols])})

    analyses.append({'id':'boxplot_summary','title':'📦 Statistical Summary — Box Plots','type':'boxplot','cols':num_cols,'score':95,'accuracy':'100%',
        'description':'Box plots showing median, spread and outliers for every sensor.',
        'why_useful':'Full statistical picture in one view. Outliers appear as dots beyond whiskers indicating abnormal readings.',
        'what_found':'Outlier analysis across all sensors.'})

    if len(num_cols) >= 2:
        strong_pairs = [(k,v) for k,v in pair_stats.items() if v['is_strongly_correlated']]
        analyses.append({'id':'correlation_heatmap','title':'🔗 Sensor Correlation Matrix','type':'heatmap','cols':num_cols,'score':97,'accuracy':'100%',
            'description':'Heatmap showing how strongly every sensor relates to every other.',
            'why_useful':'Reveals which sensors influence each other. Unexpected correlations can indicate mechanical issues.',
            'what_found':f'{len(strong_pairs)} strongly correlated pairs found.' if strong_pairs else 'No strongly correlated pairs found.'})

    trending = [(c,s) for c,s in col_stats.items() if s.get('has_trend')]
    for col, stat in trending:
        r2 = stat.get('trend_r2',0)
        analyses.append({'id':f'trend_{col}','title':f'📉 {col} Trend Analysis','type':'trend','cols':[col],'score':85+r2*10,'accuracy':f'{min(99,int(70+r2*30))}%',
            'description':f'{col} shows a statistically significant {stat.get("trend_dir","changing")} trend (R²={r2:.2f}).',
            'why_useful':f'A trend in {col} indicates a systematic change during the session — could be warm-up behavior or a warning sign.',
            'what_found':f'Changed from {stat["min"]:.1f} to {stat["max"]:.1f} with clear {stat.get("trend_dir","")} pattern.'})

    spikey = [(c,s) for c,s in col_stats.items() if s.get('has_spikes')]
    for col, stat in spikey:
        analyses.append({'id':f'spikes_{col}','title':f'⚡ {col} Spike Detection','type':'spikes','cols':[col],'score':82,'accuracy':'90%',
            'description':f'{stat["spike_count"]} significant spikes detected in {col}.',
            'why_useful':f'Sudden spikes indicate transient events — hard acceleration, sudden load changes or sensor anomalies.',
            'what_found':f'{stat["spike_count"]} spikes above threshold {stat["mean"]+2*stat["std"]:.1f}.'})

    for (c1,c2), stat in sorted(pair_stats.items(), key=lambda x: abs(x[1]['pearson_r']), reverse=True)[:8]:
        if stat['is_correlated']:
            r = stat['pearson_r']
            analyses.append({'id':f'scatter_{c1}_{c2}','title':f'🔵 {c1} vs {c2}','type':'scatter','cols':[c1,c2],'x_col':c1,'y_col':c2,'score':75+abs(r)*20,'accuracy':f'{min(98,int(70+abs(r)*30))}%',
                'description':f'{c1} and {c2} have {"strong" if abs(r)>0.6 else "moderate"} {stat["direction"]} correlation (r={r:.2f}).',
                'why_useful':f'Understanding how {c1} and {c2} relate helps diagnose engine behavior.',
                'what_found':f'Pearson r={r:.2f}, Spearman rho={stat["spearman_rho"]:.2f}. {"Nonlinear relationship detected." if stat["is_nonlinear"] else ""}'})

    high_cv = [(c,s) for c,s in col_stats.items() if s['cv'] > 0.3]
    for col, stat in high_cv:
        analyses.append({'id':f'variability_{col}','title':f'📊 {col} Variability Analysis','type':'variability','cols':[col],'score':78,'accuracy':'95%',
            'description':f'{col} shows high variability (CV={stat["cv"]:.2f}) during the session.',
            'why_useful':f'High variability in {col} means wide operating range. Expected for throttle/RPM but unexpected for temperatures.',
            'what_found':f'Min:{stat["min"]:.1f} Max:{stat["max"]:.1f} Std:{stat["std"]:.1f} Skewness:{stat["skewness"]:.2f}'})

    if len(num_cols) >= 4 and len(df) > 100:
        analyses.append({'id':'anomaly_detection','title':'🔍 Multi-Sensor Anomaly Detection','type':'anomaly','cols':num_cols,'score':90,'accuracy':'Indicative only',
            'description':f'ML (Isolation Forest) scanning all {len(num_cols)} sensors simultaneously for unusual joint readings.',
            'why_useful':'Anomalies are moments when MULTIPLE sensors deviated together — more meaningful than a single sensor spike.',
            'what_found':'By design, ~5% of readings are flagged as the most unusual in this session (relative ranking, not a fault count).'})

    if 'RPM' in num_cols and 'Driving_Mode' in df.columns:
        mode_dist = df['Driving_Mode'].value_counts(normalize=True)*100
        analyses.append({'id':'driving_mode','title':'🚦 RPM Band Analysis','type':'driving_mode','cols':['RPM'],'score':88,'accuracy':'Indicative — generic RPM bands',
            'description':'Classifies every moment by RPM band: Idle, Low, Mid, High.',
            'why_useful':'Shows time spent in each RPM range. High idle wastes fuel; sustained high-RPM increases wear and fuel use.',
            'what_found':' | '.join([f"{m}: {v:.1f}%" for m,v in mode_dist.items()])})

    stable   = [c for c,s in col_stats.items() if s['is_stable']]
    unstable = [c for c,s in col_stats.items() if not s['is_stable'] and s['cv'] > 0.1]
    if stable and unstable:
        analyses.append({'id':'stability_comparison','title':'⚖️ Sensor Stability Comparison','type':'stability','cols':num_cols,'score':80,'accuracy':'95%',
            'description':f'{len(stable)} sensors stable, {len(unstable)} highly variable.',
            'why_useful':'Sensors that should be stable but show variability may be faulty or indicate real engine instability.',
            'what_found':f'Stable: {", ".join(stable[:5])}. Variable: {", ".join(unstable[:5])}.'})

    if len(num_cols) >= 3:
        analyses.append({'id':'pca_analysis','title':'🧬 Principal Component Analysis','type':'pca','cols':num_cols,'score':76,'accuracy':'90%',
            'description':f'Reduces all {len(num_cols)} sensors into 2 dimensions to reveal hidden patterns.',
            'why_useful':'Reveals clusters of similar engine states. Outlier clusters are anomalous operating conditions.',
            'what_found':f'Analyzing {len(num_cols)} sensors to find dominant patterns.'})

    if 'MAF' in num_cols and 'Speed' in num_cols:
        analyses.append({'id':'fuel_efficiency','title':'💡 Fuel Efficiency Estimation','type':'fuel_efficiency','cols':['MAF','Speed'],'score':92,'accuracy':'85-90%',
            'description':'Estimates fuel consumption in km/l using MAF sensor formula.',
            'why_useful':'Shows exactly how fuel efficient the car was at each moment. Identifies best/worst economy conditions.',
            'what_found':'Formula: km/l = (14.7 x 6.17 x 4546 x Speed) / (3600 x MAF)'})
        analyses.append({'id':'co2_emissions','title':'🌿 CO2 Emissions Estimation','type':'co2','cols':['MAF','Speed'],'score':91,'accuracy':'85%',
            'description':'Estimates CO2 in g/km from MAF and Speed.',
            'why_useful':'CO2 is direct measure of fuel burn. <120 g/km eco, 120-180 moderate, >180 high.',
            'what_found':'Will calculate CO2 g/km for every reading.'})

    fuel_trim_cols = [c for c in ['STFT','LTFT'] if c in num_cols]
    if fuel_trim_cols:
        avg_vals = {c: df[c].mean() for c in fuel_trim_cols}
        analyses.append({'id':'fuel_trim','title':'⛽ Fuel Trim Health Analysis','type':'fuel_trim','cols':fuel_trim_cols,'score':94,'accuracy':'98%',
            'description':f'Analysis of ECU fuel corrections: {", ".join([f"{c} avg={avg_vals[c]:.1f}%" for c in fuel_trim_cols])}.',
            'why_useful':'Beyond +-10% indicates real problem — O2 sensor fault, injector issue, air leak or fuel pressure problem.',
            'what_found':' | '.join([f"{c}: {'OUT OF RANGE' if abs(avg_vals[c])>10 else 'normal'}" for c in fuel_trim_cols])})


    if 'RPM' in num_cols:
        rpm_std = df['RPM'].std()
        over_rev = (df['RPM']>4000).mean()*100
        analyses.append({'id':'driver_score','title':'🏎️ Driver Behavior Score','type':'driver_score','cols':[c for c in ['RPM','Speed','Throttle'] if c in num_cols],'score':88,'accuracy':'Composite heuristic',
            'description':'Scores driving style out of 100 based on smoothness, rev control, idle efficiency and speed stability.',
            'why_useful':'A composite indicator of driving style: higher generally means more economical and less wear on the drivetrain.',
            'what_found':f'RPM std: {rpm_std:.0f} ({"high variation" if rpm_std>500 else "low variation"}). Over-rev (>4000 RPM): {over_rev:.1f}%.'})

    analyses.sort(key=lambda x: x['score'], reverse=True)
    return analyses, col_stats, pair_stats

# ══════════════════════════════════════════════════════════════════
# EXECUTE
# ══════════════════════════════════════════════════════════════════
colors_list = ['steelblue','teal','tomato','purple','orange','green','pink','cyan','gold','coral','lime','salmon']

def execute(df, analysis):
    atype = analysis['type']
    cols  = [c for c in analysis['cols'] if c in df.columns]
    if not cols:
        st.warning("⚠️ Required columns not available")
        return False
    try:
        if atype == 'timeseries_all':
            fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, subplot_titles=cols)
            for i, col in enumerate(cols):
                fig.add_trace(go.Scatter(x=df['Time_Elapsed'], y=df[col],
                    fill='tozeroy', name=col, line=dict(color=colors_list[i%len(colors_list)])), row=i+1, col=1)
            fig.update_layout(height=max(180*len(cols),400), title='All Sensors Over Time')
            fig.update_xaxes(title_text='Time Elapsed (seconds)', row=len(cols), col=1)
            st.plotly_chart(fig, use_container_width=True)

        elif atype == 'distributions':
            grid = st.columns(3)
            for i, col in enumerate(cols):
                with grid[i%3]:
                    fig = px.histogram(df, x=col, nbins=30, title=col,
                                      color_discrete_sequence=[colors_list[i%len(colors_list)]])
                    st.plotly_chart(fig, use_container_width=True)

        elif atype == 'boxplot':
            grid = st.columns(3)
            for i, col in enumerate(cols):
                with grid[i%3]:
                    fig = px.box(df, y=col, title=col,
                                color_discrete_sequence=[colors_list[i%len(colors_list)]])
                    st.plotly_chart(fig, use_container_width=True)

        elif atype == 'heatmap':
            fig = px.imshow(df[cols].corr(), text_auto='.2f',
                           color_continuous_scale='RdBu_r', title='Sensor Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)

        elif atype == 'trend':
            col = cols[0]
            fig = px.scatter(df, x='Time_Elapsed', y=col, trendline='ols', opacity=0.4,
                            title=f'{col} Trend Over Time',
                            color_discrete_sequence=['steelblue'])
            st.plotly_chart(fig, use_container_width=True)
            slope, _, r, p, _ = stats.linregress(df['Time_Elapsed'], df[col])
            c1,c2,c3 = st.columns(3)
            c1.metric("Trend Slope",   f"{slope:.4f}/sec")
            c2.metric("R² Strength",   f"{r**2:.3f}")
            c3.metric("Direction",     "📈 Rising" if slope>0 else "📉 Falling")

        elif atype == 'spikes':
            col = cols[0]
            threshold = df[col].mean() + 2*df[col].std()
            peaks, _  = find_peaks(df[col].values, height=threshold, distance=5)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Time_Elapsed'], y=df[col],
                                    name=col, line=dict(color='steelblue', width=1)))
            if len(peaks):
                fig.add_trace(go.Scatter(x=df['Time_Elapsed'].iloc[peaks], y=df[col].iloc[peaks],
                    mode='markers', marker=dict(color='red',size=8,symbol='x'),
                    name=f'Spikes ({len(peaks)})'))
            fig.add_hline(y=threshold, line_dash='dash', line_color='orange',
                         annotation_text=f'Threshold: {threshold:.1f}')
            fig.update_layout(title=f'{col} Spike Detection')
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Total Spikes", len(peaks))

        elif atype == 'scatter':
            x_col = analysis.get('x_col', cols[0])
            y_col = analysis.get('y_col', cols[1] if len(cols)>1 else cols[0])
            if x_col in df.columns and y_col in df.columns:
                color_col = 'Driving_Mode' if 'Driving_Mode' in df.columns else None
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, opacity=0.5,
                                trendline='ols' if not color_col else None,
                                title=f'{x_col} vs {y_col}',
                                color_discrete_sequence=colors_list)
                st.plotly_chart(fig, use_container_width=True)
                r, p = stats.pearsonr(df[x_col].dropna(), df[y_col].dropna())
                c1,c2 = st.columns(2)
                c1.metric("Correlation (r)", f"{r:.3f}")
                c2.metric("Significance", "✅ Significant" if p<0.05 else "❌ Not significant")

        elif atype == 'variability':
            col = cols[0]
            fig = make_subplots(rows=2, cols=1,
                               subplot_titles=[f'{col} Over Time', f'{col} Distribution'])
            fig.add_trace(go.Scatter(x=df['Time_Elapsed'], y=df[col],
                                    name=col, line=dict(color='steelblue',width=1)), row=1, col=1)
            fig.add_hline(y=df[col].mean(), line_dash='dash', line_color='red',
                         annotation_text='Mean', row=1, col=1)
            fig.add_hline(y=df[col].mean()+df[col].std(), line_dash='dot',
                         line_color='orange', annotation_text='+1σ', row=1, col=1)
            fig.add_hline(y=df[col].mean()-df[col].std(), line_dash='dot',
                         line_color='orange', annotation_text='-1σ', row=1, col=1)
            fig.add_trace(go.Histogram(x=df[col], nbinsx=30, name='Dist',
                                      marker_color='teal'), row=2, col=1)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Mean",     f"{df[col].mean():.2f}")
            c2.metric("Std Dev",  f"{df[col].std():.2f}")
            c3.metric("Skewness", f"{df[col].skew():.2f}")
            c4.metric("Kurtosis", f"{df[col].kurtosis():.2f}")

        elif atype == 'anomaly':
            sample = df[cols].dropna()
            scaler = StandardScaler()
            scaled = scaler.fit_transform(sample)
            iso    = IsolationForest(contamination=0.05, random_state=42)
            df.loc[sample.index,'Anomaly'] = iso.fit_predict(scaled)
            normal  = df[df['Anomaly'] ==  1]
            anomaly = df[df['Anomaly'] == -1]
            c1,c2,c3 = st.columns(3)
            c1.metric("Total", f"{len(df):,}")
            c2.metric("Normal", f"{len(normal):,}")
            c3.metric("Anomalies", f"{len(anomaly):,}")
            plot_col = cols[0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=normal['Time_Elapsed'], y=normal[plot_col],
                mode='markers', marker=dict(size=2,color='steelblue'), name='Normal'))
            fig.add_trace(go.Scatter(x=anomaly['Time_Elapsed'], y=anomaly[plot_col],
                mode='markers', marker=dict(size=7,color='red',symbol='x'),
                name=f'Anomaly ({len(anomaly)})'))
            fig.update_layout(title=f'Anomalies in {plot_col}', xaxis_title='Time Elapsed (s)')
            st.plotly_chart(fig, use_container_width=True)

        elif atype == 'driving_mode':
            mode_counts = df['Driving_Mode'].value_counts().reset_index()
            mode_counts.columns = ['Mode','Count']
            c1,c2 = st.columns(2)
            with c1:
                fig = px.pie(mode_counts, names='Mode', values='Count',
                            color_discrete_sequence=['#4e79a7','#59a14f','#f28e2b','#e15759'])
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(mode_counts, x='Mode', y='Count', color='Mode',
                            color_discrete_sequence=['#4e79a7','#59a14f','#f28e2b','#e15759'])
                st.plotly_chart(fig, use_container_width=True)

        elif atype == 'stability':
            cv_data = pd.DataFrame({
                'Sensor': cols,
                'CV':     [df[c].std()/(abs(df[c].mean())+1e-9) for c in cols],
            }).sort_values('CV', ascending=False)
            fig = px.bar(cv_data, x='Sensor', y='CV',
                        title='Coefficient of Variation (higher = more variable)',
                        color='CV', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)

        elif atype == 'pca':
            sample = df[cols].dropna()
            scaler = StandardScaler()
            scaled = scaler.fit_transform(sample)
            pca    = PCA(n_components=2)
            pcs    = pca.fit_transform(scaled)
            pca_df = pd.DataFrame(pcs, columns=['PC1','PC2'])
            if 'Driving_Mode' in df.columns:
                pca_df['Driving_Mode'] = df['Driving_Mode'].values[:len(pca_df)]
            fig = px.scatter(pca_df, x='PC1', y='PC2',
                            color='Driving_Mode' if 'Driving_Mode' in pca_df.columns else None,
                            title='PCA — Engine State Clusters', opacity=0.5,
                            color_discrete_sequence=['#4e79a7','#59a14f','#f28e2b','#e15759'])
            st.plotly_chart(fig, use_container_width=True)
            var = pca.explained_variance_ratio_
            c1,c2,c3 = st.columns(3)
            c1.metric("PC1 explains", f"{var[0]*100:.1f}%")
            c2.metric("PC2 explains", f"{var[1]*100:.1f}%")
            c3.metric("Total explained", f"{sum(var)*100:.1f}%")

        # ── FUEL EFFICIENCY ─────────────────────────────────────────
        elif atype == 'fuel_efficiency':
            maf   = df['MAF'].copy().replace(0, np.nan)
            speed = df['Speed'].copy().replace(0, np.nan)
            # MAF (g/s) -> fuel mass flow (g/s) = MAF / AFR
            # -> fuel volume flow (ml/s) = fuel_mass / petrol_density
            # -> fuel volume flow (L/h)  = (ml/s) * 3600 / 1000
            # km/l = Speed (km/h) / fuel_L_per_h
            AFR             = 14.7   # stoichiometric air-fuel ratio (petrol, by mass)
            PETROL_DENSITY  = 0.745  # g/ml
            fuel_L_per_h    = (maf / (AFR * PETROL_DENSITY)) * 3.6
            fe              = speed / fuel_L_per_h
            fe    = fe.replace([np.inf, -np.inf], np.nan).clip(0, 40)
            valid = fe.dropna()

            # Always show MAF + Speed raw charts so there is never a blank screen
            fig_raw = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    subplot_titles=['MAF (g/s)', 'Speed (km/h)'])
            fig_raw.add_trace(go.Scatter(x=df['Time_Elapsed'], y=df['MAF'],
                                         name='MAF', line=dict(color='steelblue')), row=1, col=1)
            fig_raw.add_trace(go.Scatter(x=df['Time_Elapsed'], y=df['Speed'],
                                         name='Speed', line=dict(color='orange')), row=2, col=1)
            fig_raw.update_layout(height=380, title='MAF and Speed — Raw Sensor Data')
            st.plotly_chart(fig_raw, use_container_width=True)

            if len(valid) < 5:
                st.warning("⚠️ Not enough moving readings to estimate fuel efficiency "
                           "(Speed was near 0 for most of the log). "
                           "Fuel efficiency is only meaningful while the vehicle is moving.")
            else:
                avg_fe = valid.mean()
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Efficiency", f"{avg_fe:.1f} km/l")
                c2.metric("Best",           f"{valid.max():.1f} km/l")
                c3.metric("Worst",          f"{valid.min():.1f} km/l")

                plot_df = pd.DataFrame({'Time_Elapsed': df['Time_Elapsed'], 'Fuel_Efficiency': fe}).dropna()
                fig = px.line(plot_df, x='Time_Elapsed', y='Fuel_Efficiency',
                             title='Fuel Efficiency Over Time (km/l)',
                             color_discrete_sequence=['green'])
                fig.add_hline(y=avg_fe, line_dash='dash', line_color='orange',
                              annotation_text=f'Avg: {avg_fe:.1f} km/l')
                fig.update_layout(yaxis_title='km/l', xaxis_title='Time Elapsed (s)')
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.histogram(plot_df, x='Fuel_Efficiency', nbins=30,
                                    title='Fuel Efficiency Distribution',
                                    color_discrete_sequence=['green'])
                st.plotly_chart(fig2, use_container_width=True)

                if avg_fe > 12:   st.success(f"✅ Good fuel economy: {avg_fe:.1f} km/l")
                elif avg_fe > 8:  st.warning(f"⚠️ Average fuel economy: {avg_fe:.1f} km/l")
                else:             st.error(f"❌ Poor fuel economy: {avg_fe:.1f} km/l — check for rich mixture or aggressive driving")

        # ── CO2 ─────────────────────────────────────────────────────
        elif atype == 'co2':
            maf   = df['MAF'].copy().replace(0, np.nan)
            speed = df['Speed'].copy().replace(0, np.nan)
            AFR             = 14.7
            PETROL_DENSITY  = 0.745   # g/ml
            CO2_PER_LITRE   = 2310    # g CO2 per litre of petrol burned
            fuel_L_per_h    = (maf / (AFR * PETROL_DENSITY)) * 3.6
            co2             = (fuel_L_per_h * CO2_PER_LITRE) / speed
            co2   = co2.replace([np.inf, -np.inf], np.nan).clip(0, 500)
            valid = co2.dropna()

            # Always show MAF raw so screen is never blank
            fig_raw = px.line(df, x='Time_Elapsed', y='MAF',
                              title='MAF Sensor — Raw Data (used for CO2 calculation)',
                              color_discrete_sequence=['steelblue'])
            st.plotly_chart(fig_raw, use_container_width=True)

            if len(valid) < 5:
                st.warning("⚠️ Not enough moving readings for CO2 estimation "
                           "(Speed was near 0 for most of the log). "
                           "CO2 per km requires the vehicle to be moving.")
                c1, c2 = st.columns(2)
                c1.metric("MAF avg (g/s)", f"{df['MAF'].mean():.2f}")
                c2.metric("Speed avg",     f"{df['Speed'].mean():.1f} km/h")
            else:
                avg = valid.mean()
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg CO2", f"{avg:.1f} g/km")
                c2.metric("Max CO2", f"{valid.max():.1f} g/km")
                c3.metric("Min CO2", f"{valid.min():.1f} g/km")
                if avg < 120:   st.success(f"✅ CO2 low: {avg:.1f} g/km — eco driving")
                elif avg < 180: st.warning(f"⚠️ CO2 moderate: {avg:.1f} g/km")
                else:           st.error(f"❌ CO2 high: {avg:.1f} g/km — heavy fuel use")

                plot_df = pd.DataFrame({'Time_Elapsed': df['Time_Elapsed'], 'CO2': co2}).dropna()
                fig = px.line(plot_df, x='Time_Elapsed', y='CO2',
                             title='CO2 Emissions Over Time (g/km)',
                             color_discrete_sequence=['tomato'])
                fig.add_hline(y=120, line_dash='dot', line_color='green',  annotation_text='Eco limit 120')
                fig.add_hline(y=180, line_dash='dot', line_color='red',    annotation_text='High limit 180')
                fig.update_layout(yaxis_title='g/km', xaxis_title='Time Elapsed (s)')
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.histogram(plot_df, x='CO2', nbins=30,
                                    title='CO2 Distribution (g/km)',
                                    color_discrete_sequence=['tomato'])
                st.plotly_chart(fig2, use_container_width=True)

        elif atype == 'fuel_trim':
            fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, subplot_titles=cols)
            for i,col in enumerate(cols):
                fig.add_trace(go.Scatter(x=df['Time_Elapsed'],y=df[col],
                    name=col,line=dict(color='orange')),row=i+1,col=1)
                fig.add_hline(y=10,  line_dash='dash',line_color='red',row=i+1,col=1)
                fig.add_hline(y=-10, line_dash='dash',line_color='red',row=i+1,col=1)
            fig.update_layout(height=300*len(cols))
            st.plotly_chart(fig, use_container_width=True)
            for col in cols:
                avg = df[col].mean()
                if abs(avg)>10: st.warning(f"⚠️ {col} avg {avg:.1f}% — exceeds ±10%")
                else:           st.success(f"✅ {col} normal: avg {avg:.1f}%")

        elif atype == 'driver_score':
            rpm_std      = float(df['RPM'].std())
            over_rev_pct = (df['RPM']>4000).mean()*100
            # Idle = low RPM AND (effectively) not moving. Use a small speed
            # threshold instead of an exact-zero check, since speed sensors
            # rarely report exactly 0.0.
            if 'Speed' in df.columns:
                idle_pct = ((df['RPM']<800) & (df['Speed']<2)).mean()*100
            else:
                idle_pct = (df['RPM']<800).mean()*100
            speed_std    = df['Speed'].std() if 'Speed' in df.columns else 0
            scores = {
                'Smoothness':      max(0,100-rpm_std/50),
                'Rev Control':     max(0,100-over_rev_pct*2),
                'Idle Efficiency': max(0,100-idle_pct),
                'Speed Stability': max(0,100-speed_std*2),
            }
            weights = [0.3,0.3,0.2,0.2]
            overall = sum(list(scores.values())[i]*weights[i] for i in range(4))
            st.metric("🏆 Overall Driver Score", f"{overall:.1f} / 100")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Smoothness",      f"{scores['Smoothness']:.1f}")
            c2.metric("Rev Control",     f"{scores['Rev Control']:.1f}")
            c3.metric("Idle Efficiency", f"{scores['Idle Efficiency']:.1f}")
            c4.metric("Speed Stability", f"{scores['Speed Stability']:.1f}")
            fig = go.Figure(go.Scatterpolar(
                r=list(scores.values())+[list(scores.values())[0]],
                theta=list(scores.keys())+[list(scores.keys())[0]],
                fill='toself',line_color='cyan',fillcolor='rgba(0,212,255,0.2)'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
                             title='Driver Score Radar')
            st.plotly_chart(fig, use_container_width=True)

        return True
    except Exception as e:
        st.warning(f"Could not render: {e}")
        return False

# ══════════════════════════════════════════════════════════════════
# PDF — redesigned, emoji-safe, insight-rich
# ══════════════════════════════════════════════════════════════════
import re as _re

def _safe(text):
    """Strip emojis and non-latin-1 chars so fpdf never gets broken glyphs."""
    text = str(text)
    # Remove emoji / non-BMP unicode
    text = _re.sub(r'[^\x00-\xFF]', '', text)
    # Remove leftover markdown
    text = text.replace('**','').replace('*','').replace('#','')
    return text.encode('latin-1', 'replace').decode('latin-1')

# Palette
_NAVY   = (15,  35,  75)
_BLUE   = (30,  90, 180)
_LBLUE  = (210, 225, 245)
_GREEN  = (20, 140,  60)
_AMBER  = (200, 120,   0)
_RED    = (180,  20,  20)
_LGRAY  = (245, 245, 248)
_MGRAY  = (180, 180, 185)
_WHITE  = (255, 255, 255)
_BLACK  = (20,  20,  20)

class ECU_PDF(FPDF):
    # ── running header on every page except cover ─────────────────
    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*_NAVY)
        self.rect(0, 0, 210, 12, 'F')
        self.set_font('Arial', 'B', 8)
        self.set_text_color(*_WHITE)
        self.set_xy(0, 2)
        self.cell(0, 8, 'ECU Smart Analysis Report', align='C')
        self.set_text_color(*_BLACK)
        self.ln(14)

    # ── footer ────────────────────────────────────────────────────
    def footer(self):
        self.set_y(-12)
        self.set_draw_color(*_MGRAY)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_font('Arial', 'I', 7)
        self.set_text_color(*_MGRAY)
        self.cell(0, 8, f'Page {self.page_no()}  |  Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C')
        self.set_text_color(*_BLACK)

    # ── big section banner ────────────────────────────────────────
    def section(self, title):
        self.set_fill_color(*_NAVY)
        self.set_text_color(*_WHITE)
        self.set_font('Arial', 'B', 11)
        self.cell(0, 10, _safe(title), ln=True, fill=True)
        self.set_text_color(*_BLACK)
        self.ln(2)

    # ── analysis title banner ─────────────────────────────────────
    def subsection(self, title):
        self.set_fill_color(*_BLUE)
        self.set_text_color(*_WHITE)
        self.set_font('Arial', 'B', 10)
        self.cell(0, 9, _safe(title), ln=True, fill=True)
        self.set_text_color(*_BLACK)
        self.ln(2)

    # ── coloured insight label ────────────────────────────────────
    def insight_label(self, text):
        self.set_fill_color(*_LBLUE)
        self.set_text_color(*_NAVY)
        self.set_font('Arial', 'BI', 9)
        self.cell(0, 7, _safe(text), ln=True, fill=True)
        self.set_text_color(*_BLACK)

    # ── plain body paragraph ──────────────────────────────────────
    def body(self, text):
        self.set_font('Arial', '', 10)
        self.set_text_color(*_BLACK)
        self.multi_cell(0, 6, _safe(text))
        self.ln(1)

    # ── italic note ───────────────────────────────────────────────
    def note(self, text):
        self.set_font('Arial', 'I', 9)
        self.set_text_color(80, 80, 100)
        self.multi_cell(0, 5, _safe(text))
        self.set_text_color(*_BLACK)
        self.ln(1)

    # ── key-value row ─────────────────────────────────────────────
    def kv(self, key, val):
        self.set_font('Arial', 'B', 10)
        self.set_text_color(*_NAVY)
        self.cell(65, 7, _safe(key))
        self.set_font('Arial', '', 10)
        self.set_text_color(*_BLACK)
        self.cell(0, 7, _safe(val), ln=True)

    # ── metric box row (up to 4 boxes) ───────────────────────────
    def metric_row(self, items):
        """items = list of (label, value) tuples, max 4."""
        # Manual page-break check: metric_row uses raw rect()/set_xy() which
        # does NOT trigger fpdf's automatic page break, so check ourselves.
        ROW_HEIGHT = 22
        if self.get_y() + ROW_HEIGHT > self.page_break_trigger:
            self.add_page()

        n    = len(items)
        w    = (self.w - 20) / n
        x0   = self.get_x()
        y0   = self.get_y()
        for i, (label, value) in enumerate(items):
            x = 10 + i * w
            # box background
            self.set_fill_color(*_LGRAY)
            self.set_draw_color(*_MGRAY)
            self.rect(x, y0, w - 2, 18, 'FD')
            # value
            self.set_font('Arial', 'B', 12)
            self.set_text_color(*_NAVY)
            self.set_xy(x, y0 + 2)
            self.cell(w - 2, 7, _safe(str(value)), align='C')
            # label
            self.set_font('Arial', '', 8)
            self.set_text_color(80, 80, 100)
            self.set_xy(x, y0 + 10)
            self.cell(w - 2, 5, _safe(label), align='C')
        self.set_text_color(*_BLACK)
        self.set_xy(10, y0 + ROW_HEIGHT)

    # ── verdict badge ─────────────────────────────────────────────
    def verdict(self, status, text):
        cfg = {
            'ok':       (_GREEN, _WHITE, 'GOOD'),
            'warn':     (_AMBER, _WHITE, 'WARNING'),
            'critical': (_RED,   _WHITE, 'CRITICAL'),
        }
        bg, fg, badge = cfg.get(status, (_BLACK, _WHITE, 'NOTE'))
        # Badge pill
        self.set_fill_color(*bg)
        self.set_text_color(*fg)
        self.set_font('Arial', 'B', 9)
        self.cell(22, 7, badge, fill=True, align='C')
        # Text
        self.set_fill_color(*_LGRAY)
        self.set_text_color(*_BLACK)
        self.set_font('Arial', '', 9)
        remaining = self.w - self.get_x() - 10
        self.cell(remaining, 7, _safe(text), fill=True, ln=True)
        self.ln(2)

    # ── thin horizontal rule ──────────────────────────────────────
    def rule(self):
        self.set_draw_color(*_MGRAY)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)


def _summarize_analysis(pdf, df, a):
    """Write data-driven narrative + insights for each analysis type."""
    atype = a['type']
    cols  = [c for c in a['cols'] if c in df.columns]
    if not cols:
        pdf.body('Required columns not available in this dataset.')
        return

    # ── TIME SERIES ───────────────────────────────────────────────
    if atype == 'timeseries_all':
        duration = df['Time_Elapsed'].max() if 'Time_Elapsed' in df.columns else len(df)
        pdf.body(
            f"The session logged {len(df):,} readings across {len(cols)} sensors "
            f"over {duration:.0f} seconds ({duration/60:.1f} minutes)."
        )
        pdf.rule()
        pdf.insight_label('Sensor Range Summary')
        wide = []
        for c in cols:
            rng  = df[c].max() - df[c].min()
            mean = df[c].mean()
            cv   = df[c].std() / (abs(mean) + 1e-9)
            pdf.body(f"  {c}: {df[c].min():.1f} - {df[c].max():.1f}  (mean {mean:.1f}, CV {cv:.2f})")
            if mean > 0 and rng / mean > 1.0:
                wide.append(c)
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        if wide:
            pdf.body(
                f"{', '.join(wide)} showed very wide swings relative to their mean values. "
                f"This is normal for driver-controlled sensors (throttle, RPM) but if you see "
                f"it on temperature or voltage sensors it can indicate instability or a fault."
            )
        else:
            pdf.body(
                "All sensors stayed within moderate ranges throughout the session, "
                "which indicates consistent, stable engine operation with no extreme events."
            )

    # ── DISTRIBUTIONS ─────────────────────────────────────────────
    elif atype == 'distributions':
        pdf.insight_label('Distribution Shape Per Sensor')
        for c in cols:
            skew = df[c].skew()
            kurt = df[c].kurtosis()
            if abs(skew) > 1:
                shape = f"heavily {'right' if skew>0 else 'left'}-skewed (skew={skew:.2f})"
            elif kurt > 3:
                shape = f"heavy-tailed (kurtosis={kurt:.2f})"
            else:
                shape = "roughly normal"
            pdf.body(f"  {c}: avg={df[c].mean():.2f}, std={df[c].std():.2f}, "
                     f"range [{df[c].min():.2f} - {df[c].max():.2f}]  --  {shape}.")
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        skewed = [c for c in cols if abs(df[c].skew()) > 1]
        if skewed:
            pdf.body(
                f"{', '.join(skewed)} have skewed distributions, meaning the engine spent "
                f"most of its time at one end of the operating range. A right-skewed RPM "
                f"distribution (most time at low RPM) typically indicates city or idle-heavy "
                f"driving. A left-skewed temperature (most time at high temp) could flag a "
                f"cooling concern worth investigating."
            )
        else:
            pdf.body(
                "All sensors show roughly symmetric distributions. The engine operated across "
                "a balanced range with no extreme bias toward one end, which is a healthy sign."
            )

    # ── BOX PLOTS ─────────────────────────────────────────────────
    elif atype == 'boxplot':
        pdf.insight_label('Outlier Count Per Sensor')
        any_outliers = False
        for c in cols:
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr  = q3 - q1
            out  = df[(df[c] < q1-1.5*iqr) | (df[c] > q3+1.5*iqr)]
            pct  = len(out)/len(df)*100
            stat = 'ok' if pct<2 else ('warn' if pct<8 else 'critical')
            pdf.verdict(stat,
                f"{c}: {len(out)} outliers ({pct:.1f}%)  |  median={df[c].median():.2f}  IQR={iqr:.2f}")
            if pct >= 2:
                any_outliers = True
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        if any_outliers:
            pdf.body(
                "Sensors with more than 2% outlier readings experienced brief abnormal spikes "
                "or dips. A small number of outliers is normal — engine events like gear changes, "
                "sudden acceleration or brief load spikes all show up here. If outliers cluster "
                "at a specific time in the log, they likely share a common cause worth examining."
            )
        else:
            pdf.body(
                "All sensors stayed well within their expected value ranges with very few outliers. "
                "This indicates smooth, consistent operation with no sudden abnormal events."
            )

    # ── CORRELATION HEATMAP ───────────────────────────────────────
    elif atype == 'heatmap':
        strong = []
        for c1, c2 in itertools.combinations(cols, 2):
            try:
                r, p = stats.pearsonr(df[c1].dropna(), df[c2].dropna())
                if abs(r) > 0.6 and p < 0.05:
                    strong.append((c1, c2, r))
            except: pass
        pdf.insight_label('Strongly Correlated Sensor Pairs  (|r| > 0.6)')
        if strong:
            for c1, c2, r in sorted(strong, key=lambda x: abs(x[2]), reverse=True)[:10]:
                direction = "positive" if r > 0 else "negative"
                pdf.body(f"  {c1} vs {c2}: r={r:.3f}  ({direction})")
        else:
            pdf.body("  No strongly correlated pairs found.")
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        if strong:
            pos = [(a,b,r) for a,b,r in strong if r>0]
            neg = [(a,b,r) for a,b,r in strong if r<0]
            if pos:
                pdf.body(
                    f"Positive correlations (e.g. {pos[0][0]} and {pos[0][1]}) mean these "
                    f"sensors rise and fall together — typical for load-dependent sensors like "
                    f"RPM, throttle, MAF and engine load. This is expected and healthy."
                )
            if neg:
                pdf.body(
                    f"Negative correlations (e.g. {neg[0][0]} and {neg[0][1]}) mean one sensor "
                    f"rises as the other falls. This can be normal (e.g. efficiency drops as RPM "
                    f"rises) but unexpected inverse pairs are worth investigating as they may "
                    f"indicate a sensor fault or unusual engine behaviour."
                )
        else:
            pdf.body(
                "No strong correlations found. This is common when the dataset only contains a "
                "few sensors, or when the drive was short. It does not indicate a problem."
            )

    # ── TREND ─────────────────────────────────────────────────────
    elif atype == 'trend':
        c = cols[0]
        slope, intercept, r, p, _ = stats.linregress(df['Time_Elapsed'], df[c])
        direction   = "increasing" if slope > 0 else "decreasing"
        total_change = slope * df['Time_Elapsed'].max()
        start_val   = df[c].iloc[:10].mean()
        end_val     = df[c].iloc[-10:].mean()
        pdf.metric_row([
            ('Start Value', f"{start_val:.1f}"),
            ('End Value',   f"{end_val:.1f}"),
            ('Total Change',f"{total_change:+.2f}"),
            ('R2',          f"{r**2:.3f}"),
        ])
        pdf.body(
            f"{c} showed a statistically significant {direction} trend "
            f"(slope={slope:.4f}/sec, R2={r**2:.3f}, p={p:.4f}). "
            f"Over the {df['Time_Elapsed'].max():.0f}-second session it changed by {total_change:.2f} units."
        )
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        if 'Temp' in c or 'temp' in c.lower():
            if slope > 0 and end_val < 105:
                pdf.verdict('ok',
                    f"{c} is warming up normally. Reaching operating temperature helps fuel "
                    f"atomisation and reduces engine wear.")
            elif slope > 0 and end_val >= 105:
                pdf.verdict('warn',
                    f"{c} is still rising at the end of the log and has reached {end_val:.1f}. "
                    f"Check coolant level and thermostat if this continues.")
            else:
                pdf.verdict('warn',
                    f"{c} is falling during the session. This can be normal after warm-up "
                    f"but a sharp drop may indicate thermostat or cooling system issues.")
        elif 'Speed' in c:
            pdf.body(
                f"Speed trending {direction} across the session. "
                f"{'Likely the vehicle was decelerating or approaching a stop at the end of the log.' if slope<0 else 'Vehicle was progressively accelerating or moving to higher-speed roads.'}"
            )
            pdf.verdict('ok', "Speed trend is informational — no fault indicated.")
        else:
            pdf.body(
                f"A persistent {direction} trend in {c} can indicate a real gradual change "
                f"(warm-up, load increase) or sensor drift. If the trend continues beyond this "
                f"session it is worth logging again to confirm the pattern."
            )
            status = 'ok' if abs(r**2) < 0.6 else 'warn'
            pdf.verdict(status,
                "Weak trend — likely normal session variation." if status=='ok'
                else f"Strong trend (R2={r**2:.2f}) — {c} is systematically changing. Monitor closely.")

    # ── SPIKES ────────────────────────────────────────────────────
    elif atype == 'spikes':
        c         = cols[0]
        threshold = df[c].mean() + 2 * df[c].std()
        peaks, _  = find_peaks(df[c].values, height=threshold, distance=5)
        spike_times = ', '.join([f"{df['Time_Elapsed'].iloc[p]:.1f}s" for p in peaks[:8]])
        suffix      = '...' if len(peaks) > 8 else ''
        pdf.metric_row([
            ('Total Spikes', str(len(peaks))),
            ('Threshold',    f"{threshold:.1f}"),
            ('Mean',         f"{df[c].mean():.1f}"),
            ('Max',          f"{df[c].max():.1f}"),
        ])
        pdf.body(
            f"{c} recorded {len(peaks)} spikes above {threshold:.1f} "
            f"(mean + 2 standard deviations). "
            f"Spike times: {spike_times}{suffix}."
        )
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        if c == 'RPM':
            pdf.body(
                "RPM spikes represent sudden engine speed surges — these happen during hard "
                "acceleration, gear changes, or rev-matching. A small number is perfectly normal. "
                "Many spikes close together can indicate aggressive driving or clutch slip."
            )
        elif 'Temp' in c:
            pdf.body(
                "Temperature spikes are unusual and worth investigating. Brief spikes could be "
                "sensor noise or a loose connection. Repeated spikes may indicate a cooling "
                "system issue such as a failing thermostat, air pockets in the coolant, or "
                "an intermittently blocked radiator."
            )
        elif c in ['STFT','LTFT']:
            pdf.body(
                "Fuel trim spikes mean the ECU was making large, sudden corrections to the "
                "air-fuel mixture. This can be caused by brief fuel pressure drops, O2 sensor "
                "glitches, or momentary air leaks. Frequent spikes suggest an underlying issue."
            )
        else:
            pdf.body(
                f"Spikes in {c} represent brief abnormal readings. Occasional spikes are "
                f"expected during transient events (hard acceleration, gear shifts, load changes). "
                f"If spikes are clustered or frequent, consider checking the sensor connection "
                f"and the relevant system for faults."
            )
        status = 'ok' if len(peaks)<5 else ('warn' if len(peaks)<20 else 'critical')
        pdf.verdict(status,
            "Low spike count — normal transient events only." if len(peaks)<5
            else ("Moderate spike count — investigate cause if not aggressive driving." if len(peaks)<20
            else "High spike count — frequent abnormal events detected. Check sensor and system health."))

    # ── SCATTER ───────────────────────────────────────────────────
    elif atype == 'scatter':
        c1, c2 = cols[0], cols[1]
        r, p   = stats.pearsonr(df[c1].dropna(), df[c2].dropna())
        rho, _ = stats.spearmanr(df[c1].dropna(), df[c2].dropna())
        strength  = "strong" if abs(r)>0.6 else ("moderate" if abs(r)>0.3 else "weak")
        direction = "positive" if r>0 else "negative"
        nonlinear = abs(rho) > abs(r) + 0.15
        pdf.metric_row([
            ('Pearson r',    f"{r:.3f}"),
            ('Spearman rho', f"{rho:.3f}"),
            ('p-value',      f"{p:.4f}"),
            ('Relationship', 'Nonlinear' if nonlinear else 'Linear'),
        ])
        pdf.body(
            f"{c1} and {c2} show a {strength} {direction} correlation (r={r:.3f}). "
            f"{'A nonlinear pattern is present — the sensors influence each other in a curved way.' if nonlinear else 'The relationship is approximately linear.'}"
        )
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        pair = {c1, c2}
        if 'RPM' in pair and 'Speed' in pair:
            pdf.body(
                f"RPM vs Speed correlation of {r:.2f} reflects your gear usage. "
                f"A high correlation means consistent gear selection. A weak correlation "
                f"suggests frequent gear changes, slip, or CVT-style operation."
            )
        elif 'RPM' in pair and 'Engine_Load' in pair:
            pdf.body(
                f"RPM and Engine Load have a {strength} relationship. High load at low RPM "
                f"(lugging) increases wear and fuel consumption. High RPM at low load is "
                f"typically fine but wastes fuel if sustained."
            )
        elif 'MAF' in pair and 'RPM' in pair:
            pdf.body(
                f"MAF and RPM should correlate positively — more air enters as the engine "
                f"revs higher. A weak or negative correlation here can indicate a faulty MAF "
                f"sensor, air intake restriction, or boost leak on turbocharged engines."
            )
        elif 'STFT' in pair or 'LTFT' in pair:
            pdf.body(
                f"Fuel trim correlating with another sensor reveals what is driving the ECU's "
                f"fuel corrections. If trim correlates strongly with load or RPM, the mixture "
                f"error is load-dependent — typical of injector wear or a vacuum leak that "
                f"worsens under load."
            )
        else:
            pdf.body(
                f"A {strength} {direction} relationship between {c1} and {c2} is {'expected and confirms normal engine dynamics.' if abs(r)>0.5 else 'weak, suggesting these systems operate largely independently under these conditions.'}"
            )
        status = 'ok' if abs(r)<0.95 else 'warn'
        pdf.verdict(status,
            "Correlation is in expected range." if status=='ok'
            else "Very high correlation (r > 0.95) — sensors may be redundant or directly derived from each other.")

    # ── VARIABILITY ───────────────────────────────────────────────
    elif atype == 'variability':
        c    = cols[0]
        cv   = df[c].std() / (abs(df[c].mean()) + 1e-9)
        skew = df[c].skew()
        pdf.metric_row([
            ('Mean',     f"{df[c].mean():.2f}"),
            ('Std Dev',  f"{df[c].std():.2f}"),
            ('CV',       f"{cv:.3f}"),
            ('Skewness', f"{skew:.2f}"),
        ])
        pdf.body(
            f"{c} has a coefficient of variation (CV) of {cv:.3f}. "
            f"Range: {df[c].min():.2f} to {df[c].max():.2f}."
        )
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        expected_variable = c in ['RPM','Throttle','Speed','Accel_Pedal','Engine_Load']
        if cv > 0.5 and not expected_variable:
            pdf.body(
                f"{c} is showing high variability (CV={cv:.2f}) for a sensor that should be "
                f"relatively stable. This pattern can indicate: a failing sensor, intermittent "
                f"electrical connection, or a real mechanical instability in the system "
                f"{c} measures. Compare against a known-good log to confirm."
            )
            pdf.verdict('warn', f"Unexpectedly high variability in {c} — inspect sensor and wiring.")
        elif cv > 0.5 and expected_variable:
            pdf.body(
                f"{c} is a dynamic, driver-controlled sensor so high variability (CV={cv:.2f}) "
                f"is expected. It simply reflects varying driving conditions — acceleration, "
                f"braking, and load changes all appear here."
            )
            pdf.verdict('ok', f"Variability in {c} is normal for a driver-input sensor.")
        elif cv > 0.3:
            pdf.body(
                f"Moderate variability (CV={cv:.2f}). The engine was operating across a wide "
                f"but not extreme range. This is typical of mixed city and highway driving."
            )
            pdf.verdict('ok', "Moderate variability — consistent with mixed driving conditions.")
        else:
            pdf.body(
                f"{c} was very stable throughout the session (CV={cv:.2f}). "
                f"For temperature sensors this confirms the engine reached and held operating "
                f"temperature. For control sensors it means steady-state driving with minimal input."
            )
            pdf.verdict('ok', f"{c} is stable and consistent — no anomalies.")

    # ── ANOMALY ───────────────────────────────────────────────────
    elif atype == 'anomaly':
        sample    = df[cols].dropna()
        scaler    = StandardScaler()
        scaled    = scaler.fit_transform(sample)
        iso       = IsolationForest(contamination=0.05, random_state=42)
        labels    = iso.fit_predict(scaled)
        n_anomaly = (labels == -1).sum()
        pct       = n_anomaly / len(labels) * 100
        pdf.metric_row([
            ('Total Readings', f"{len(labels):,}"),
            ('Normal',         f"{(labels==1).sum():,}"),
            ('Anomalies',      f"{n_anomaly:,}"),
            ('Anomaly Rate',   f"{pct:.1f}%"),
        ])
        pdf.body(
            f"Isolation Forest scanned {len(cols)} sensors simultaneously across "
            f"{len(sample):,} readings. It flagged {n_anomaly} anomalous moments ({pct:.1f}%) "
            f"where multiple sensors deviated jointly from their normal patterns."
        )
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        pdf.body(
            "Multi-sensor anomalies are more meaningful than single-sensor spikes because they "
            "require several readings to be unusual at the same time. Common causes include: "
            "hard acceleration events, sudden braking, gear shifts, sensor glitches, or brief "
            "mechanical events like a misfire. The dashboard chart shows the exact timestamps "
            "of each anomaly — reviewing those moments in the raw data can pinpoint the cause."
        )
        status = 'ok' if pct<6 else ('warn' if pct<12 else 'critical')
        pdf.verdict(status,
            "Anomaly rate within expected range — no systemic issues detected." if pct<6
            else ("Elevated anomaly rate — review flagged timestamps in the dashboard." if pct<12
            else "High anomaly rate — significant abnormal operation detected. Investigate urgently."))

    # ── DRIVING MODE ─────────────────────────────────────────────
    elif atype == 'driving_mode':
        mode_dist = df['Driving_Mode'].value_counts(normalize=True) * 100
        pdf.insight_label('Time Spent in Each RPM Band')
        for mode, pct in mode_dist.items():
            mode_s = str(mode)
            status = 'warn' if (mode_s.startswith('High RPM') and pct>15) or (mode_s.startswith('Idle') and pct>30) else 'ok'
            if mode_s.startswith('Idle'):
                comment = 'High idle — burns fuel with no progress.' if pct>30 else 'Low idle — efficient.'
            elif mode_s.startswith('Low RPM'):
                comment = 'Dominant low-RPM operation — typical of city/stop-go driving.' if pct>40 else 'Some low-RPM operation.'
            elif mode_s.startswith('Mid RPM'):
                comment = 'Good mid-range share — efficient cruising band for most engines.' if pct>30 else 'Limited mid-range operation.'
            else:  # High RPM
                comment = 'Significant high-RPM operation — check if intentional (e.g. towing, sport driving).' if pct>15 else 'Minimal high-RPM operation — good for engine longevity.'
            pdf.verdict(status, f"{mode}: {pct:.1f}%  |  {comment}")
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        idle_pct = float(sum(v for k,v in mode_dist.items() if str(k).startswith('Idle')))
        high_pct = float(sum(v for k,v in mode_dist.items() if str(k).startswith('High RPM')))
        mid_pct  = float(sum(v for k,v in mode_dist.items() if str(k).startswith('Mid RPM')))
        if high_pct > 15:
            pdf.body(
                f"With {high_pct:.1f}% of the session above 4000 RPM, engine wear and "
                f"fuel consumption are elevated. Sustained high-RPM operation generates more heat "
                f"and stresses drivetrain components faster. If this wasn't intentional (e.g. "
                f"overtaking, towing), it's worth checking gear-shift timing."
            )
        elif idle_pct > 30:
            pdf.body(
                f"The engine spent {idle_pct:.1f}% of the session below 800 RPM. Note this RPM "
                f"band can include both true idle (stationary) and very low-RPM cruising in tall "
                f"gears — check the dashboard's RPM vs Speed chart to distinguish the two. "
                f"If this represents genuine idling, extended idling wastes fuel and can dilute "
                f"engine oil with unburnt fuel over time."
            )
        elif mid_pct > 40:
            pdf.body(
                f"{mid_pct:.1f}% of the session was in the 2000-4000 RPM band — typically the "
                f"most fuel-efficient operating range for most engines under load. This is a "
                f"healthy distribution for sustained driving."
            )
        else:
            pdf.body(
                "The RPM distribution shows a typical mixed-use profile. Fuel consumption and "
                "engine wear are in a normal range for this type of usage. Note: these RPM bands "
                "are generic defaults and not tuned to this specific vehicle's gearing."
            )

    # ── STABILITY ────────────────────────────────────────────────
    elif atype == 'stability':
        pdf.insight_label('Sensor Stability Ranking (CV — lower is more stable)')
        unstable = []
        for c in sorted(cols, key=lambda x: df[x].std()/(abs(df[x].mean())+1e-9), reverse=True):
            cv = df[c].std() / (abs(df[c].mean()) + 1e-9)
            stable = cv < 0.05
            status = 'ok' if stable else ('warn' if cv<0.3 else 'critical')
            pdf.verdict(status, f"{c}: CV={cv:.3f}  --  {'STABLE' if stable else ('VARIABLE' if cv<0.3 else 'HIGHLY VARIABLE')}")
            if not stable and c not in ['RPM','Throttle','Speed','Accel_Pedal']:
                unstable.append((c, cv))
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        if unstable:
            pdf.body(
                f"The following sensors showed unexpected variability: "
                f"{', '.join([f'{c} (CV={v:.2f})' for c,v in unstable])}. "
                f"Sensors that should be stable (temperatures, battery voltage, fuel pressure) "
                f"showing high CV values can indicate: a degrading sensor, wiring issues, "
                f"vacuum leaks, or real instability in the system being measured."
            )
            pdf.verdict('warn', "Investigate unexpectedly variable sensors before the next service.")
        else:
            pdf.body(
                "All sensors that are expected to be stable are reading consistently. "
                "This confirms good sensor health and stable engine operating conditions."
            )
            pdf.verdict('ok', "Sensor stability is good across all channels.")

    # ── PCA ───────────────────────────────────────────────────────
    elif atype == 'pca':
        sample = df[cols].dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(sample)
        pca    = PCA(n_components=min(2, len(cols)))
        pca.fit(scaled)
        var    = pca.explained_variance_ratio_
        pdf.metric_row([
            ('Sensors analysed', str(len(cols))),
            ('PC1 explains',     f"{var[0]*100:.1f}%"),
            ('PC2 explains',     f"{var[1]*100:.1f}%" if len(var)>1 else 'N/A'),
            ('Total captured',   f"{sum(var)*100:.1f}%"),
        ])
        pdf.body(
            f"PCA reduced {len(cols)} sensors to 2 dimensions. "
            f"Together PC1 and PC2 capture {sum(var)*100:.1f}% of the total variance in the dataset."
        )
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        if sum(var) > 0.75:
            pdf.body(
                f"High explained variance ({sum(var)*100:.1f}%) means most engine behaviour "
                f"in this session can be described by just one or two dominant patterns — "
                f"likely a speed/load axis and a temperature axis. This is typical of "
                f"straightforward driving sessions and indicates the sensors are measuring "
                f"a coherent, predictable system."
            )
            pdf.verdict('ok', "Engine behaviour is well-structured and predictable.")
        else:
            pdf.body(
                f"Lower explained variance ({sum(var)*100:.1f}%) means engine behaviour is "
                f"complex and spread across many independent dimensions. This is common in "
                f"varied driving sessions or when sensors cover very different systems. "
                f"It is not a fault indicator by itself."
            )
            pdf.verdict('ok', "Complex multi-dimensional behaviour — normal for varied driving.")

    # ── FUEL EFFICIENCY ──────────────────────────────────────────
    elif atype == 'fuel_efficiency':
        maf   = df['MAF'].copy().replace(0, np.nan)
        speed = df['Speed'].copy().replace(0, np.nan)
        AFR             = 14.7
        PETROL_DENSITY  = 0.745
        fuel_L_per_h    = (maf / (AFR * PETROL_DENSITY)) * 3.6
        fe              = speed / fuel_L_per_h
        fe    = fe.replace([np.inf,-np.inf], np.nan).clip(0, 40)
        valid = fe.dropna()
        if len(valid) < 5:
            pdf.body(
                "Insufficient moving data to calculate fuel efficiency. "
                f"MAF average: {df['MAF'].mean():.2f} g/s, Speed average: {df['Speed'].mean():.1f} km/h. "
                "The vehicle was mostly stationary during this log."
            )
            pdf.verdict('warn', "Fuel efficiency could not be calculated — log is mostly stationary.")
        else:
            avg, best, worst = valid.mean(), valid.max(), valid.min()
            pdf.metric_row([
                ('Average',   f"{avg:.1f} km/l"),
                ('Best',      f"{best:.1f} km/l"),
                ('Worst',     f"{worst:.1f} km/l"),
                ('Std Dev',   f"{valid.std():.1f}"),
            ])
            pdf.body(
                f"Estimated fuel efficiency averaged {avg:.1f} km/l. "
                f"Best economy ({best:.1f} km/l) occurred at steady cruising speed. "
                f"Worst ({worst:.1f} km/l) occurred under hard acceleration or low-speed high-load conditions."
            )
            pdf.rule()
            pdf.insight_label('What This Means for Your Vehicle')
            if avg > 15:
                pdf.body(
                    f"Excellent fuel economy at {avg:.1f} km/l. This typically reflects steady "
                    f"highway driving, a well-tuned engine, and conservative throttle use. "
                    f"Keep tyre pressures correct and air filter clean to maintain this level."
                )
            elif avg > 10:
                pdf.body(
                    f"Good fuel economy at {avg:.1f} km/l. Mixed driving conditions are reflected "
                    f"in the wide best-to-worst range. Economy can be improved by smoother "
                    f"acceleration, earlier gear shifts, and avoiding unnecessary idling."
                )
            elif avg > 7:
                pdf.body(
                    f"Average fuel economy at {avg:.1f} km/l. City driving, heavy traffic, "
                    f"or an engine running slightly rich are the most common causes. "
                    f"Check fuel trims (STFT/LTFT), tyre pressures and air filter condition."
                )
            else:
                pdf.body(
                    f"Poor fuel economy at {avg:.1f} km/l. This indicates the engine may be "
                    f"running rich, there may be injector fouling, a faulty O2 or MAF sensor, "
                    f"or the vehicle was being driven very aggressively. A full diagnostic scan is recommended."
                )
            status = 'ok' if avg>10 else ('warn' if avg>7 else 'critical')
            pdf.verdict(status,
                "Fuel economy is good." if avg>10
                else ("Fuel economy is below average — investigate." if avg>7
                else "Poor fuel economy — diagnostic scan recommended."))

    # ── CO2 ───────────────────────────────────────────────────────
    elif atype == 'co2':
        maf   = df['MAF'].copy().replace(0, np.nan)
        speed = df['Speed'].copy().replace(0, np.nan)
        AFR             = 14.7
        PETROL_DENSITY  = 0.745
        CO2_PER_LITRE   = 2310
        fuel_L_per_h    = (maf / (AFR * PETROL_DENSITY)) * 3.6
        co2             = (fuel_L_per_h * CO2_PER_LITRE) / speed
        co2   = co2.replace([np.inf,-np.inf], np.nan).clip(0, 500)
        valid = co2.dropna()
        if len(valid) < 5:
            pdf.body("Insufficient moving data for CO2 estimation.")
            pdf.verdict('warn', "CO2 could not be calculated — vehicle was mostly stationary.")
        else:
            avg = valid.mean()
            pdf.metric_row([
                ('Average CO2', f"{avg:.1f} g/km"),
                ('Max CO2',     f"{valid.max():.1f} g/km"),
                ('Min CO2',     f"{valid.min():.1f} g/km"),
                ('EU Eco limit','120 g/km'),
            ])
            pdf.body(
                f"Estimated CO2 averaged {avg:.1f} g/km. The EU eco benchmark is <120 g/km; "
                f">180 g/km is considered high."
            )
            pdf.rule()
            pdf.insight_label('What This Means for Your Vehicle')
            if avg < 120:
                pdf.body(
                    f"CO2 at {avg:.1f} g/km is within the eco range. This vehicle is operating "
                    f"efficiently and would likely pass modern emissions standards. "
                    f"Maintaining good fuel economy habits will keep emissions low."
                )
            elif avg < 180:
                pdf.body(
                    f"CO2 at {avg:.1f} g/km is in the moderate range. This is typical for "
                    f"larger-engined vehicles or city driving. Smoother driving, correct "
                    f"tyre pressures, and timely servicing can reduce this further."
                )
            else:
                pdf.body(
                    f"CO2 at {avg:.1f} g/km is high. Possible causes: engine running rich, "
                    f"faulty injectors, worn O2 sensor, aggressive driving, or a heavy load. "
                    f"A diagnostic check focusing on fuelling and emissions systems is advised."
                )
            status = 'ok' if avg<120 else ('warn' if avg<180 else 'critical')
            pdf.verdict(status,
                "CO2 is within eco range." if avg<120
                else ("CO2 is moderate." if avg<180
                else "High CO2 — fuelling system check recommended."))

    # ── FUEL TRIM ─────────────────────────────────────────────────
    elif atype == 'fuel_trim':
        pdf.insight_label('Fuel Trim Health Per Channel')
        any_fault = False
        for c in cols:
            avg     = df[c].mean()
            pct_out = (df[c].abs()>10).mean()*100
            status  = 'ok' if abs(avg)<=5 else ('warn' if abs(avg)<=10 else 'critical')
            pdf.verdict(status,
                f"{c}: avg={avg:+.1f}%  |  {pct_out:.1f}% of readings outside +-10%  "
                f"|  range [{df[c].min():.1f}% to {df[c].max():.1f}%]")
            if abs(avg) > 5:
                any_fault = True
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        pdf.body(
            "Fuel trim is the percentage correction the ECU applies to the injector pulse width "
            "to maintain the target air-fuel ratio. A healthy engine needs less than +-5% "
            "correction. Values between +-5% and +-10% indicate the system is compensating "
            "for something. Values beyond +-10% mean a real problem exists."
        )
        stft_avg = df['STFT'].mean() if 'STFT' in cols else None
        ltft_avg = df['LTFT'].mean() if 'LTFT' in cols else None
        if stft_avg is not None and ltft_avg is not None:
            if stft_avg > 0 and ltft_avg > 0:
                pdf.body(
                    "Both STFT and LTFT are positive — the ECU is adding fuel. "
                    "The engine is running lean. Common causes: vacuum/air leak, weak fuel pump, "
                    "clogged injectors, or a failing MAF sensor reading low."
                )
            elif stft_avg < 0 and ltft_avg < 0:
                pdf.body(
                    "Both STFT and LTFT are negative — the ECU is removing fuel. "
                    "The engine is running rich. Common causes: leaking injector, high fuel "
                    "pressure, faulty MAP/MAF sensor reading high, or a coolant temp sensor "
                    "telling the ECU the engine is still cold."
                )
        if any_fault:
            pdf.verdict('warn', "Fuel trim outside healthy range — a diagnostic scan with live O2 data is recommended.")
        else:
            pdf.verdict('ok', "Fuel trim healthy — ECU making only minor corrections.")

    # ── DRIVER SCORE ─────────────────────────────────────────────
    elif atype == 'driver_score':
        rpm_std      = float(df['RPM'].std())
        over_rev_pct = (df['RPM']>4000).mean()*100
        if 'Speed' in df.columns:
            idle_pct = ((df['RPM']<800) & (df['Speed']<2)).mean()*100
        else:
            idle_pct = (df['RPM']<800).mean()*100
        speed_std    = df['Speed'].std() if 'Speed' in df.columns else 0
        scores = {
            'Smoothness':      max(0,100-rpm_std/50),
            'Rev Control':     max(0,100-over_rev_pct*2),
            'Idle Efficiency': max(0,100-idle_pct),
            'Speed Stability': max(0,100-speed_std*2),
        }
        overall = sum(list(scores.values())[i]*[0.3,0.3,0.2,0.2][i] for i in range(4))
        pdf.metric_row([
            ('Overall Score',    f"{overall:.1f}/100"),
            ('Smoothness',       f"{scores['Smoothness']:.1f}"),
            ('Rev Control',      f"{scores['Rev Control']:.1f}"),
            ('Speed Stability',  f"{scores['Speed Stability']:.1f}"),
        ])
        pdf.body(
            f"Overall driver score: {overall:.1f}/100. "
            f"Smoothness {scores['Smoothness']:.1f} (RPM std={rpm_std:.0f}). "
            f"Rev Control {scores['Rev Control']:.1f} ({over_rev_pct:.1f}% over 4000 RPM). "
            f"Idle Efficiency {scores['Idle Efficiency']:.1f} ({idle_pct:.1f}% unnecessary idle). "
            f"Speed Stability {scores['Speed Stability']:.1f} (speed std={speed_std:.1f} km/h)."
        )
        pdf.rule()
        pdf.insight_label('What This Means for Your Vehicle')
        worst_dim = min(scores, key=scores.get)
        advice = {
            'Smoothness':
                f"Smoothness is the lowest sub-score ({scores['Smoothness']:.1f}). "
                f"High RPM variation (std={rpm_std:.0f}) suggests frequent hard acceleration "
                f"and braking. Smoother inputs reduce fuel use by up to 15-20% and extend "
                f"clutch, brake and drivetrain life significantly.",
            'Rev Control':
                f"Rev Control is the weakest area ({scores['Rev Control']:.1f}). "
                f"{over_rev_pct:.1f}% of the session was above 4000 RPM. "
                f"Upshifting earlier (around 2000-2500 RPM for petrol, 1500-2000 for diesel) "
                f"reduces fuel consumption, engine wear, and noise.",
            'Idle Efficiency':
                f"Idle Efficiency needs improvement ({scores['Idle Efficiency']:.1f}). "
                f"The engine spent {idle_pct:.1f}% of the session idling without moving. "
                f"Turning off the engine for stops over 60 seconds saves fuel and reduces "
                f"exhaust system loading.",
            'Speed Stability':
                f"Speed Stability is the weakest area ({scores['Speed Stability']:.1f}). "
                f"Speed varied by +-{speed_std:.1f} km/h on average. "
                f"More consistent throttle use on open roads improves fuel economy and "
                f"reduces drivetrain stress.",
        }
        pdf.body(advice.get(worst_dim, ''))
        status = 'ok' if overall>70 else ('warn' if overall>45 else 'critical')
        pdf.verdict(status,
            "Good driving behaviour — smooth and economical." if overall>70
            else ("Average driving — some habits to improve for better economy and less wear." if overall>45
            else "Aggressive driving style — significantly increases fuel cost, wear and safety risk."))
    else:
        pdf.body(a.get('what_found', 'No summary available.'))


def build_pdf(df, analyses_done, vehicle_name):
    pdf = ECU_PDF()

    # ── COVER PAGE ────────────────────────────────────────────────
    pdf.add_page()
    # Dark header band
    pdf.set_fill_color(*_NAVY)
    pdf.rect(0, 0, 210, 60, 'F')
    pdf.set_font('Arial', 'B', 26)
    pdf.set_text_color(*_WHITE)
    pdf.set_xy(0, 14)
    pdf.cell(0, 12, 'ECU SMART ANALYSIS', align='C', ln=True)
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, 'Vehicle Diagnostic Report', align='C', ln=True)
    pdf.set_text_color(*_BLACK)

    # Vehicle info block
    pdf.set_xy(30, 72)
    pdf.set_fill_color(*_LGRAY)
    pdf.set_draw_color(*_MGRAY)
    pdf.rect(25, 68, 160, 42, 'FD')
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(*_NAVY)
    pdf.set_xy(30, 72)
    pdf.cell(0, 8, _safe(vehicle_name or 'Unknown Vehicle'), ln=True)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(*_BLACK)
    pdf.set_x(30)
    pdf.cell(0, 7, f"Date: {datetime.now().strftime('%d %B %Y  %H:%M')}", ln=True)
    if 'Time_Elapsed' in df.columns:
        dur = df['Time_Elapsed'].max()
        pdf.set_x(30)
        pdf.cell(0, 7, f"Session duration: {dur:.0f} s  ({dur/60:.1f} min)  |  {len(df):,} readings", ln=True)
    num_cols_cover = [c for c in df.select_dtypes(include=np.number).columns
                      if c not in ['Time_Elapsed','Anomaly','RUL','NOx','CO2','Fuel_Efficiency','MAF_safe']
                      and not c.startswith('IS_')]
    pdf.set_x(30)
    pdf.cell(0, 7, f"Sensors: {', '.join(num_cols_cover)}", ln=True)

    # Summary metric row
    pdf.set_xy(10, 120)
    pdf.metric_row([
        ('Analyses run',   str(len(analyses_done))),
        ('Sensors logged', str(len(num_cols_cover))),
        ('Total readings', f"{len(df):,}"),
        ('Session length', f"{df['Time_Elapsed'].max():.0f}s" if 'Time_Elapsed' in df.columns else 'N/A'),
    ])

    # Quick-verdict summary
    pdf.ln(6)
    pdf.section('QUICK FINDINGS OVERVIEW')
    for a in analyses_done:
        title_clean = _safe(a['title'])
        pdf.set_font('Arial', 'B', 9)
        pdf.set_text_color(*_NAVY)
        pdf.cell(90, 6, title_clean)
        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(*_BLACK)
        pdf.cell(0, 6, _safe(f"Accuracy: {a['accuracy']}  |  Confidence: {a['score']:.0f}/100"), ln=True)
    pdf.ln(4)

    # ── DRIVE SUMMARY PAGE ────────────────────────────────────────
    pdf.add_page()
    pdf.section('DRIVE SUMMARY — SENSOR STATISTICS')
    pdf.set_font('Arial', '', 10)
    for col in num_cols_cover:
        try:
            pdf.metric_row([
                (f"{col} avg",  f"{df[col].mean():.2f}"),
                ('max',         f"{df[col].max():.2f}"),
                ('min',         f"{df[col].min():.2f}"),
                ('std dev',     f"{df[col].std():.2f}"),
            ])
        except: pass
    pdf.ln(4)

    # ── PER-ANALYSIS PAGES ────────────────────────────────────────
    for a in analyses_done:
        pdf.add_page()
        pdf.subsection(_safe(a['title']))
        pdf.note(f"Accuracy: {a['accuracy']}  |  Confidence: {a['score']:.0f}/100  |  {_safe(a['description'])}")
        pdf.ln(1)
        try:
            _summarize_analysis(pdf, df, a)
        except Exception as e:
            pdf.body(f"Could not generate detailed summary for this analysis ({e}).")
            pdf.body(a.get('what_found', ''))
        pdf.ln(3)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(tmp.name)
    return tmp.name

# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════
st.title("🚗 ECU Smart Analysis Dashboard")
st.caption("Upload any ECU CSV — statistical brain generates custom analysis automatically")

uploaded = st.file_uploader("📂 Upload your ECU CSV file", type=['csv','txt'])

if uploaded:
    # ── Detect new file by name+size and reset all state ──────────
    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.uploaded_file_id != file_id:
        for key in ['df','analyses','col_stats','pair_stats','selected']:
            st.session_state[key] = None
        st.session_state.ran               = False
        st.session_state.uploaded_file_id  = file_id

    if st.session_state.df is None:
        with st.spinner("Reading and standardizing file..."):
            raw_df, fmt = parse_file(uploaded)
            df          = standardize(raw_df.copy())
        st.session_state.df  = df
        st.session_state.fmt = fmt
    else:
        df  = st.session_state.df
        fmt = st.session_state.get('fmt','CSV')

    st.success(f"✅ **{fmt}** — {len(df):,} rows, {len(df.columns)} columns")

    st.subheader("📄 Full Data Preview")
    st.dataframe(df, use_container_width=True, height=300)

    st.subheader("📊 Data Overview")
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
               if c not in ['Time_Elapsed'] and not c.startswith('IS_')]
    overview = pd.DataFrame({
        'Sensor':  num_cols,
        'Min':     [round(df[c].min(),2) for c in num_cols],
        'Max':     [round(df[c].max(),2) for c in num_cols],
        'Average': [round(df[c].mean(),2) for c in num_cols],
        'Std Dev': [round(df[c].std(),2) for c in num_cols],
        'Missing': [int(df[c].isnull().sum()) for c in num_cols],
    })
    st.dataframe(overview, use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Rows",     f"{len(df):,}")
    c2.metric("Total Columns",  f"{len(df.columns)}")
    c3.metric("Sensor Columns", f"{len(num_cols)}")
    c4.metric("Duration",       f"{df['Time_Elapsed'].max():.0f}s" if 'Time_Elapsed' in df.columns else "N/A")

    vehicle_name = st.text_input("🚗 Vehicle Name (optional)", placeholder="e.g. Honda City 2019")

    st.divider()

    if st.button("🧠 Analyse My Data", type="primary"):
        with st.spinner("🔍 Running statistical analysis..."):
            analyses, col_stats, pair_stats = generate_analysis_plan(df)
        st.session_state.analyses   = analyses
        st.session_state.col_stats  = col_stats
        st.session_state.pair_stats = pair_stats
        st.session_state.ran        = False

    if st.session_state.analyses:
        analyses   = st.session_state.analyses
        col_stats  = st.session_state.col_stats
        pair_stats = st.session_state.pair_stats

        st.success(f"✅ Brain found **{len(analyses)} analyses** for your data!")

        strong = sorted(pair_stats.items(), key=lambda x: abs(x[1]['pearson_r']), reverse=True)[:6]
        if strong:
            with st.expander("🔗 Strongest Sensor Relationships Found", expanded=False):
                rel_data = pd.DataFrame([{
                    'Sensor 1':    p[0][0],
                    'Sensor 2':    p[0][1],
                    'Correlation': f"{p[1]['pearson_r']:.3f}",
                    'Strength':    'Strong' if abs(p[1]['pearson_r'])>0.6 else 'Moderate',
                    'Direction':   p[1]['direction'].capitalize(),
                    'Nonlinear':   '✅' if p[1]['is_nonlinear'] else '❌',
                } for p in strong])
                st.dataframe(rel_data, use_container_width=True)

        st.subheader("✅ Select Analyses to Run")
        selected = []
        cols_ui  = st.columns(2)
        for i, a in enumerate(analyses):
            with cols_ui[i%2]:
                if st.checkbox(
                    f"{a['title']}  \n*Accuracy: {a['accuracy']} | Score: {a['score']:.0f}/100*",
                    value=True, key=f"sel_{a['id']}"):
                    selected.append(a)

        if st.button("▶️ Run Selected Analyses", type="primary"):
            st.session_state.selected      = selected
            st.session_state.ran           = True
            st.session_state.vehicle_name  = vehicle_name

    if st.session_state.ran and st.session_state.get('selected'):
        selected     = st.session_state.selected
        vehicle_name = st.session_state.get('vehicle_name','')
        df           = st.session_state.df
        analyses_done= []

        st.divider()
        st.subheader("📊 Analysis Results")

        for a in selected:
            st.subheader(a['title'])
            with st.expander("ℹ️ What is this and why is it useful?", expanded=False):
                st.markdown(f"**📌 What it does:** {a['description']}")
                st.markdown(f"**💡 Why useful:** {a['why_useful']}")
                st.markdown(f"**🔎 What brain found:** {a['what_found']}")
                st.markdown(f"**📊 Accuracy:** {a['accuracy']}  |  **Score:** {a['score']:.0f}/100")
                st.markdown(f"**🔧 Columns:** {', '.join(a['cols'])}")

            result = execute(df, a)
            if result:
                analyses_done.append(a)
            st.divider()

        st.subheader("📥 Download Reports")
        c1, c2 = st.columns(2)

        with c1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📊 Download Cleaned CSV", csv,
                              file_name="cleaned_ecu.csv", mime="text/csv")

        with c2:
            if analyses_done:
                with st.spinner("Building PDF report..."):
                    pdf_path = build_pdf(df, analyses_done, vehicle_name)
                with open(pdf_path,'rb') as f:
                    pdf_bytes = f.read()
                os.unlink(pdf_path)
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"ecu_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    key="pdf_download"
                )
                st.success("✅ PDF ready — click above to download!")
            else:
                st.info("No completed analyses to include in PDF yet.")