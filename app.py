#%%
import pandas as pd

# קריאת הקובץ
df = pd.read_csv("alarms.csv")

# המרת העמודה time ל-datetime
df["time"] = pd.to_datetime(df["time"])

df = df.drop(columns=["threat"])

print(df.head())

#%%
print(df.columns)

#%%
import pandas as pd
coords = pd.read_csv("coord.csv")  # columns: loc, lat, long
coords = coords.drop_duplicates(subset="loc")


#%%

# Quick check
print(coords.head())

# Merge with main DataFrame
# Assuming df["origin"] matches coords["loc"]
df = df.merge(coords, left_on="cities", right_on="loc", how="left")

# Drop 'loc' column from merge if you want, since it's the same as 'origin'
df.drop(columns=["loc"], inplace=True)

print(df.head())

#%%
# כמה null יש בכל עמודה
df.isnull().sum()

#%%
# ערכים ייחודיים בעמודת description
print("Unique values in description:")
print(df["description"].unique())

print("\n" + "="*50 + "\n")

# ערכים ייחודיים בעמודת origin
print("Unique values in origin:")
print(df["origin"].unique())

#%%
# ספירה של כל הערכים בעמודת description
description_counts = df["description"].value_counts()

print("ספירה של כל סוגי ההתראות:")
print(description_counts)

#%%
# ספירה של כל הערכים בעמודת description
origin_counts = df["origin"].value_counts()

print("ספירה של כל סוגי המדינות:")
print(origin_counts)

#%%
# סינון רשומות שמקורן בישראל
israel_alerts = df[df["origin"] == "Israel"]

# ספירה לפי description
israel_counts = israel_alerts["description"].value_counts()

print("סוגי ההתראות עבור origin = Israel:")
print(israel_counts)

#%%
# סינון רעידות אדמה
earthquakes = df[df["description"] == "רעידת אדמה"]

# ספירה לפי מקור
earthquake_origins = earthquakes["origin"].value_counts()

print("רעידות אדמה לפי מקור:")
print(earthquake_origins)

#%%
# Filter only for 'חדירת מחבלים'
infiltration_alerts = df[df["description"] == 'חדירת מחבלים']

# Count by origin
infiltration_by_origin = infiltration_alerts["origin"].value_counts()

print("Hostile Infiltration alerts by origin:")
print(infiltration_by_origin)

#%%
target_descriptions = ['ירי רקטות וטילים', 'חדירת כלי טיס עוין']

# 2. סינון ה-DataFrame
# הפעולה אומרת: "תשאיר רק שורות שבהן הערך בעמודת description נמצא ברשימה שלנו"
df = df[df["description"].isin(target_descriptions)].copy()

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import webbrowser
from threading import Timer

# --- 1. טעינת וניקוי נתונים ---
# וודא שה-DataFrame 'df' קיים
# df = pd.read_csv(r"C:\path\to\your\data.csv")

df.columns = df.columns.str.strip()
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['hour'] = df['time'].dt.hour
df['date'] = df['time'].dt.date

for col in ['description', 'origin', 'cities']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

relevant_origins = ["Iran", "Gaza", "Lebanon", "Yemen", "Iraq", "Syria"]
df_filtered = df[df['origin'].isin(relevant_origins)].copy()

alert_types = sorted([t for t in df_filtered['description'].unique() if t.lower() != 'nan'])
origin_types = sorted([o for o in df_filtered['origin'].unique() if o.lower() != 'nan'])

min_date = df_filtered['date'].min()
max_date = df_filtered['date'].max()

# --- 2. בניית האפליקציה ---
# הוספת external_stylesheets יכולה לפעמים לעזור לסידור ה-CSS, אך כאן נפתור זאת ידנית
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("מפת איומים אינטראקטיבית", style={'textAlign': 'center', 'fontFamily': 'Arial', 'color': '#333'}),

    html.Div([
        # שורה 1: תאריכים ושעות
        html.Div([
            html.Div([
                html.Label("📅 טווח תאריכים:", style={'fontWeight': 'bold'}),

                # === התיקון הגדול ללוח השנה ===
                # אנו עוטפים את הלוח בדיב שמיישר אותו לשמאל (LTR) וצובע את הטקסט בשחור
                # זה מונע את הריצודים וההעלמות של הימים
                html.Div([
                    dcc.DatePickerRange(
                        id='date-picker',
                        min_date_allowed=min_date, max_date_allowed=max_date,
                        start_date=min_date, end_date=max_date,
                        display_format='DD/MM/YYYY',
                        # מבטיח שהלוח יצוף מעל הכל
                        style={'zIndex': 9999, 'position': 'relative'}
                    )
                ], style={'direction': 'ltr', 'textAlign': 'left', 'color': 'black'})
                # ^^^ סוף התיקון ^^^

            ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),  # verticalAlign עוזר ליישור

            html.Div([
                html.Label("⏰ שעות ביממה:", style={'fontWeight': 'bold'}),
                dcc.RangeSlider(id='hour-slider', min=0, max=23, step=1,
                                marks={i: f'{i}:00' for i in range(0, 25, 3)}, value=[0, 23])
            ], style={'width': '50%', 'display': 'inline-block', 'float': 'right', 'direction': 'rtl'}),
            # הסליידר נשאר מימין לשמאל
        ], style={'marginBottom': '30px', 'zIndex': 1000, 'position': 'relative'}),

        # שורה 2: פילטרים
        html.Div([
            html.Div([
                html.Label("🛡️ סוג התרעה:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='desc-filter',
                    options=[{'label': t, 'value': t} for t in alert_types],
                    value=alert_types,
                    multi=True,
                    placeholder="בחר סוגי איום..."
                )
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                html.Label("🌍 מדינת מקור:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='orig-filter',
                    options=[{'label': o, 'value': o} for o in origin_types],
                    value=origin_types,
                    multi=True,
                    placeholder="בחר מדינות..."
                )
            ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        ])

    ], style={'padding': '25px', 'backgroundColor': '#f8f9fa', 'borderRadius': '15px',
              'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'direction': 'rtl'}),

    # גרף
    dcc.Graph(id='word-cloud-graph', style={'height': '75vh', 'marginTop': '20px'})
], style={'padding': '20px', 'fontFamily': 'Arial'})


# --- 3. לוגיקה ---
@app.callback(
    Output('word-cloud-graph', 'figure'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('hour-slider', 'value'),
     Input('desc-filter', 'value'),
     Input('orig-filter', 'value')]
)
def update_graph(start_date, end_date, hour_range, selected_desc, selected_orig):
    # וידוא תקינות תאריכים (למקרה של בחירה חלקית)
    if not start_date or not end_date:
        return go.Figure().update_layout(title="נא לבחור תאריכים")

    mask = (
            (df_filtered['date'] >= pd.to_datetime(start_date).date()) &
            (df_filtered['date'] <= pd.to_datetime(end_date).date()) &
            (df_filtered['hour'] >= hour_range[0]) &
            (df_filtered['hour'] <= hour_range[1])
    )
    temp_df = df_filtered[mask].copy()

    if selected_desc:
        temp_df = temp_df[temp_df['description'].isin(selected_desc)]
    else:
        temp_df = temp_df[0:0]

    if selected_orig:
        temp_df = temp_df[temp_df['origin'].isin(selected_orig)]
    else:
        temp_df = temp_df[0:0]

    city_counts = temp_df['cities'].value_counts().reset_index()
    city_counts.columns = ['city_name', 'count']

    if city_counts.empty:
        return go.Figure().update_layout(title="לא נמצאו נתונים", title_x=0.5)

    max_c = city_counts['count'].max()
    min_c = city_counts['count'].min()

    def get_color(cnt):
        if max_c == min_c: return "rgb(255, 0, 0)"
        ratio = (cnt - min_c) / (max_c - min_c)
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        return f'rgb({r}, {g}, 0)'

    np.random.seed(42)
    city_counts['x'] = np.random.uniform(0, 100, len(city_counts))
    city_counts['y'] = np.random.uniform(0, 100, len(city_counts))

    fig = go.Figure(data=[go.Scatter(
        x=city_counts['x'],
        y=city_counts['y'],
        mode='text',
        text=city_counts['city_name'],
        hovertext=city_counts['count'].apply(lambda x: f"מספר אזעקות: {x}"),
        textfont=dict(
            size=np.interp(city_counts['count'], (min_c, max_c), (12, 90)),
            color=[get_color(c) for c in city_counts['count']],
            family="Arial, sans-serif",
            weight="bold"
        )
    )])

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        title=f"סה\"כ אזעקות בחתך הנבחר: {len(temp_df):,}",
        title_x=0.5,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


# --- 4. הרצה בפורט 8055 ---
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8055/")


if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=False, port=8055)
