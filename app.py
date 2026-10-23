from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import ACTION_NAMES, Maze, aggregate_dynamic, aggregate_static


st.set_page_config(
    page_title="Работа 7 | Планирование",
    page_icon="7",
    layout="wide",
    initial_sidebar_state="auto",
)

COLORS = {
    "dyna_q": "#136F63",
    "dyna_q_plus": "#E76F51",
    "action_bonus": "#D4A72C",
    "prioritized": "#315C8C",
    "trajectory": "#7A5195",
}
LABELS = {
    "dyna_q": "Dyna-Q",
    "dyna_q_plus": "Dyna-Q+",
    "action_bonus": "Бонус при выборе",
    "prioritized": "Приоритетный проход",
    "trajectory": "Траекторная выборка",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root { --navy:#132238; --ink:#182230; --teal:#136F63; --line:#DDE3EA; }
        .stApp { background:#FFFFFF; color:var(--ink); }
        [data-testid="stSidebar"] { background:var(--navy); border-right:0; }
        [data-testid="stSidebar"] * { color:#F7FAFC; }
        [data-testid="stSidebar"] [data-baseweb="select"] * { color:#182230; }
        [data-testid="stSidebar"] input { color:#182230; }
        h1,h2,h3 { letter-spacing:0 !important; color:var(--ink); }
        h1 { font-size:2.1rem !important; margin-bottom:.25rem !important; }
        h2 { font-size:1.35rem !important; margin-top:1.25rem !important; }
        .meta { color:#5B6777; font-size:.92rem; margin-bottom:1.5rem; }
        .theory { border-left:4px solid var(--teal); padding:.25rem 1rem; margin:.8rem 0 1.2rem; }
        .theory p { margin:.35rem 0; }
        div[data-testid="stMetric"] { border-top:2px solid var(--line); padding-top:.7rem; }
        div[data-testid="stButton"] button { border-radius:6px; border:1px solid #0F5A51; background:#136F63; color:white; }
        div[data-testid="stButton"] button:hover { background:#0F5A51; border-color:#0F5A51; color:white; }
        [data-testid="stDataFrame"] { border:1px solid var(--line); }
        footer { visibility:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(
        f'<div class="meta">Баранов Вадим · № 2 по списку · {subtitle}</div>',
        unsafe_allow_html=True,
    )


def theory(text: str) -> None:
    st.markdown(f'<div class="theory">{text}</div>', unsafe_allow_html=True)


def line_figure(series: dict[str, np.ndarray], y_title: str) -> go.Figure:
    fig = go.Figure()
    fallback = ["#136F63", "#E76F51", "#315C8C", "#D4A72C", "#7A5195"]
    for index, (label, values) in enumerate(series.items()):
        color = COLORS.get(label, fallback[index % len(fallback)])
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(values) + 1),
                y=values,
                mode="lines",
                name=LABELS.get(label, label),
                line={"width": 2.2, "color": color},
            )
        )
    fig.update_layout(
        template="plotly_white",
        font={"color": "#182230"},
        height=470,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Шаг взаимодействия" if y_title != "Шагов в эпизоде" else "Эпизод",
        yaxis_title=y_title,
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#E7EBF0", zeroline=False)
    fig.update_yaxes(gridcolor="#E7EBF0", zeroline=False)
    return fig


def maze_figure(maze: Maze, path: list[tuple[int, int]], step: int) -> go.Figure:
    grid = np.zeros((maze.height, maze.width))
    for x, y in maze.walls(step):
        grid[y, x] = 1
    colorscale = [[0.0, "#F5F7FA"], [0.49, "#F5F7FA"], [0.5, "#27364A"], [1, "#27364A"]]
    fig = go.Figure(go.Heatmap(z=grid, colorscale=colorscale, showscale=False, zmin=0, zmax=1))
    if path:
        fig.add_trace(
            go.Scatter(
                x=[point[0] for point in path],
                y=[point[1] for point in path],
                mode="lines+markers",
                line={"color": "#E76F51", "width": 3},
                marker={"size": 7},
                name="Жадный маршрут",
            )
        )
    fig.add_annotation(x=maze.start[0], y=maze.start[1], text="S", showarrow=False, font={"color": "#136F63", "size": 17})
    fig.add_annotation(x=maze.goal[0], y=maze.goal[1], text="G", showarrow=False, font={"color": "#E76F51", "size": 17})
    fig.update_layout(
        template="plotly_white",
        font={"color": "#182230"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=360,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={
            "dtick": 1,
            "side": "top",
            "range": [-0.5, 8.5],
            "constrain": "domain",
            "fixedrange": True,
        },
        yaxis={
            "dtick": 1,
            "range": [5.5, -0.5],
            "scaleanchor": "x",
            "scaleratio": 1,
            "constrain": "domain",
            "fixedrange": True,
        },
        showlegend=False,
    )
    return fig


@st.cache_data(show_spinner=False)
def cached_static(episodes: int, runs: int) -> dict[str, np.ndarray]:
    return aggregate_static(episodes, runs)


@st.cache_data(show_spinner=False)
def cached_dynamic(
    algorithms: tuple[str, ...], scenario: str, planning_steps: int, runs: int
) -> dict:
    return aggregate_dynamic(algorithms, scenario, planning_steps, runs)


inject_css()

with st.sidebar:
    st.markdown("## Работа 7")
    st.caption("Планирование и модели")
    page = st.radio(
        "Раздел",
        (
            "О работе",
            "1. Dyna и n-шаговый метод",
            "2. Изменяющиеся лабиринты",
            "3. Приоритеты и траектории",
            "4. Теоретические выводы",
        ),
    )
    st.divider()
    st.caption("Исполнитель")
    st.write("Баранов Вадим")
    st.write("Вариант: 2")

if page == "О работе":
    header("Планирование в обучении с подкреплением", "полный отчёт по работе 7")
    theory(
        "<b>Dyna</b> объединяет прямое обучение на реальном переходе, обучение модели среды "
        "и планирование по синтетическому опыту. Один реальный шаг поэтому может породить "
        "несколько полезных обновлений Q-функции."
    )
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.subheader("Что выполнено")
        st.markdown(
            """
            - сравнение Dyna-Q с одношаговым и n-шаговым обучением;
            - воспроизведение blocking maze и shortcut maze;
            - реализация Dyna-Q+ по Саттону и Барто;
            - вариант с бонусом только на этапе выбора действия;
            - приоритетный проход и траекторная выборка;
            - ответы по стохастической и нестационарной средам.
            """
        )
        st.subheader("Схема одного шага Dyna")
        st.code(
            "реальный переход → TD-обновление → обновление модели\n"
            "                  ↓\n"
            "        n модельных обновлений Q",
            language="text",
        )
    with right:
        st.plotly_chart(
            maze_figure(Maze("static"), [], 0), theme=None, width="stretch"
        )
        st.caption("Статический лабиринт из примера № 1. S — старт, G — цель.")

elif page == "1. Dyna и n-шаговый метод":
    header("Dyna против n-шагового SARSA", "обсуждение 1")
    theory(
        "n-шаговый SARSA передаёт награду максимум на <i>n</i> переходов за одно обновление. "
        "Dyna-Q повторно использует уже изученную модель и за то же число реальных шагов "
        "распространяет информацию существенно дальше."
    )
    controls, chart = st.columns([0.28, 0.72], gap="large")
    with controls:
        episodes = st.slider("Эпизоды", 20, 100, 50, 10)
        runs = st.slider("Независимые запуски", 2, 20, 5)
        st.caption("Показывается среднее число шагов до цели; меньше — лучше.")
    with chart:
        with st.spinner("Сравниваю методы…"):
            values = cached_static(episodes, runs)
        st.plotly_chart(
            line_figure(values, "Шагов в эпизоде"), theme=None, width="stretch"
        )

    summary = pd.DataFrame(
        {
            "Метод": list(values),
            "Среднее, последние 10 эпизодов": [
                float(np.mean(curve[-10:])) for curve in values.values()
            ],
            "Лучший эпизод": [float(np.min(curve)) for curve in values.values()],
        }
    ).sort_values("Среднее, последние 10 эпизодов")
    st.dataframe(summary, hide_index=True, width="stretch")
    st.subheader("Ответ")
    st.write(
        "В этом лабиринте n-шаговый метод обычно улучшает одношаговый вариант, но не достигает "
        "скорости Dyna-Q с 5–50 шагами планирования. Ограниченный n-шаговый возврат использует "
        "только текущую траекторию; Dyna многократно обновляет любые ранее встреченные пары (s, a)."
    )

elif page == "2. Изменяющиеся лабиринты":
    header("Dyna-Q+ в нестационарной среде", "обсуждение 2, задания 1–3")
    theory(
        "Dyna-Q+ добавляет к модельной награде бонус <b>κ√τ</b>, где τ — время с последней "
        "проверки пары (s, a). Поэтому давно не проверенные действия сохраняют шанс быть исследованными."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        scenario_label = st.selectbox(
            "Сценарий",
            ("Блокирующий лабиринт", "Открытие короткого пути"),
        )
    with c2:
        planning_steps = st.select_slider("Шагов планирования n", (0, 5, 10, 20, 50), value=10)
    with c3:
        runs = st.slider("Запусков для усреднения", 2, 20, 4, key="dynamic_runs")
    scenario = "blocking" if scenario_label.startswith("Блок") else "shortcut"
    algorithms = ("dyna_q", "dyna_q_plus", "action_bonus")
    with st.spinner("Запускаю серию экспериментов…"):
        result = cached_dynamic(algorithms, scenario, planning_steps, runs)

    left, right = st.columns([0.68, 0.32], gap="large")
    with left:
        st.plotly_chart(
            line_figure({key: value["mean"] for key, value in result.items()}, "Накопленная награда"),
            theme=None,
            width="stretch",
        )
    with right:
        best_key = max(result, key=lambda key: result[key]["final_reward"])
        st.plotly_chart(
            maze_figure(Maze(scenario), result[best_key]["path"], Maze(scenario).default_horizon),
            theme=None,
            width="stretch",
        )
        st.caption(f"Финальный жадный маршрут: {LABELS[best_key]}.")

    change_step = Maze(scenario).change_step or 0
    metrics = pd.DataFrame(
        [
            {
                "Метод": LABELS[key],
                "Целей к концу": round(float(value["final_reward"]), 2),
                "До изменения": round(float(value["mean"][change_step - 1]), 2),
                "После изменения": round(float(value["final_reward"] - value["mean"][change_step - 1]), 2),
            }
            for key, value in result.items()
        ]
    )
    st.dataframe(metrics, hide_index=True, width="stretch")

    st.subheader("Интерпретация")
    st.markdown(
        """
        1. **Почему Dyna-Q+ лучше.** Бонус заставляет перепроверять давно не выбранные действия. После изменения стены агент быстрее обнаруживает новый выход или короткий путь.
        2. **Почему разрыв в первой части shortcut maze меньше.** До открытия короткого пути среда стационарна. Дополнительное исследование Dyna-Q+ ещё не даёт преимущества и расходует часть шагов.
        3. **Бонус только при выборе.** Он не искажает TD-цели, поэтому оценки отражают исходную награду. На графике вариант обычно стабильнее, но информация о новом пути распространяется медленнее: бонус направляет реальные действия, а не каждое модельное обновление.
        """
    )

elif page == "3. Приоритеты и траектории":
    header("Направленное планирование", "обсуждения 3 и 5")
    theory(
        "Приоритетный проход выбирает модельные обновления с большой абсолютной TD-ошибкой. "
        "Траекторная выборка вместо независимых пар разворачивает модель вперёд из текущего состояния."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        scenario_label = st.selectbox(
            "Сценарий",
            ("Блокирующий лабиринт", "Открытие короткого пути"),
            key="directed_scenario",
        )
    with c2:
        planning_steps = st.select_slider(
            "Бюджет планирования", (5, 10, 20, 50), value=10, key="directed_n"
        )
    with c3:
        runs = st.slider("Запусков", 2, 15, 4, key="directed_runs")
    scenario = "blocking" if scenario_label.startswith("Блок") else "shortcut"
    algorithms = ("dyna_q", "dyna_q_plus", "prioritized", "trajectory")
    with st.spinner("Сравниваю способы выборки обновлений…"):
        result = cached_dynamic(algorithms, scenario, planning_steps, runs)
    st.plotly_chart(
        line_figure({key: value["mean"] for key, value in result.items()}, "Накопленная награда"),
        theme=None,
        width="stretch",
    )
    ranking = pd.DataFrame(
        {
            "Метод": [LABELS[key] for key in algorithms],
            "Финальная награда": [result[key]["final_reward"] for key in algorithms],
            "Длина найденного пути": [max(0, len(result[key]["path"]) - 1) for key in algorithms],
        }
    ).sort_values("Финальная награда", ascending=False)
    st.dataframe(ranking, hide_index=True, width="stretch")
    st.write(
        "Приоритетный проход эффективен, когда небольшое число неожиданных переходов должно быстро "
        "повлиять на предшествующие состояния. Траекторная выборка лучше согласует планирование с "
        "распределением состояний текущей политики, но может прекратить развёртку при неизвестном переходе."
    )

else:
    header("Выводы и ответы", "обсуждения 2 и 4")
    st.subheader("Стохастическая среда")
    theory(
        "Детерминированная модель Dyna хранит один исход для (s, a). Для стохастической среды "
        "нужно хранить распределение или набор наблюдавшихся (r, s′), оценивать частоты и при "
        "планировании выбирать исход из этого распределения. Ожидаемое обновление использует "
        "взвешенную сумму по всем исходам."
    )
    st.subheader("Почему одной модификации недостаточно")
    st.write(
        "В нестационарной среде старые наблюдения продолжают доминировать в накопленных частотах. "
        "Модель медленно забывает прежнюю динамику, а чрезмерное постоянное исследование расходует "
        "реальные шаги. Практическое решение — скользящее окно или экспоненциальное забывание, "
        "детектор изменения TD-ошибки и адаптивный уровень исследования."
    )
    st.subheader("Выборочные и полные обновления")
    st.write(
        "При сильно несимметричном распределении преимущество выборочных обновлений возрастает. "
        "Они естественно тратят вычисления пропорционально вероятности исходов; полное обновление "
        "каждый раз перечисляет и редкие переходы, которые почти не влияют на математическое ожидание."
    )
    st.subheader("Итог")
    st.markdown(
        """
        - Dyna-Q выигрывает за счёт повторного использования модели, а не более длинного реального опыта.
        - Dyna-Q+ платит небольшую цену в стационарном режиме, но быстрее адаптируется после изменения.
        - Бонус при выборе сохраняет чистые оценки Q, тогда как бонус в планировании быстрее распространяет сигнал исследования.
        - Приоритеты полезны для редких больших ошибок; траектории — для обновлений, близких к текущей политике.
        """
    )

st.divider()
st.caption("Работа 7 · Планирование · Баранов Вадим · список № 2")
