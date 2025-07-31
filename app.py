import random
import os
import json
import itertools
import pandas as pd
from pathlib import Path
import gradio as gr
import openai
import numpy as np
import altair as alt

PROMPT_TEMPLATE = """
Ты выступаешь в роли профессионального макроэкономиста, эксперта по экономике России, денежно-кредитной политике Банка России и финансовым рынкам.
Перед тобой представлены два варианта прогноза ключевых макроэкономических показателей России на конец 2026 и 2027 годов.
Каждый из аналитиков предоставил прогнозные значения в формате JSON, включающие следующие показатели:

CPI (индекс потребительских цен)

GDP (темпы роста ВВП)

KeyRate (ключевая ставка Банка России)

ExchangeRate (курс рубля к доллару США)

UralsPrice (цена нефти марки Urals)

Кроме того, каждый аналитик приложил подробное текстовое обоснование своего прогноза, включающее описание сценариев, предпосылок, макроэкономических условий и причинно-следственных связей.

Твоя задача — определить, какой из двух представленных прогнозов более реалистичен, аргументирован и соответствует текущим и ожидаемым макроэкономическим реалиям России. При принятии решения опирайся на следующие критерии:

реалистичность и обоснованность предпосылок;

логичность причинно-следственных связей в обосновании;

соответствие актуальным и ожидаемым тенденциям экономики России (инфляция, курс рубля, цена на нефть, политика ЦБ РФ).

Для проверки предположений и обоснований, при необходимости, ты можешь использовать интернет для поиска и уточнения актуальной информации по экономике и рынкам. Самостоятельно решай, какие именно данные требуют дополнительной проверки.

Вот два варианта прогнозов и обоснований:

Прогноз аналитика №1:
{model_1}

Прогноз аналитика №2:
{model_2}

После тщательного анализа выбери ТОЛЬКО ОДИН, наиболее реалистичный и аргументированный прогноз.

Ответом должна быть только одна цифра:

1 — если более реалистичен и обоснован прогноз аналитика №1;

2 — если более реалистичен и обоснован прогноз аналитика №2.

Больше никаких комментариев, пояснений или дополнительного текста в ответе быть не должно.
"""

GAUNTLET_PROMPT_TEMPLATE = """
  Ты — профессиональный макроэкономист и финансовый аналитик Центрального банка России.
  Твоя задача — сформировать реалистичный экспертный прогноз развития российской экономики
  на конец 2026 и 2027 годов по следующим показателям:

  • Инфляция («CPI»)
  • Реальный ВВП («GDP»)
  • Ключевая ставка («KeyRate»)
  • Курс рубля к доллару («ExchangeRate»)
  • Цена нефти марки Urals («UralsPrice»)

  ---

  ### Входные данные

  1. **Базовый сценарий ДКП** — таблица с целевыми значениями ключевых показателей:
    {baseline}

  2. **Зафиксированные дискретные сценарии ДКП**
    (каждый характеризуется целыми значениями «спрос» и «предложение» в диапазоне [–1; 1]):
    {scenarios}

  3. **Саммари новостного фона** (с момента последнего обновления статистики) по категориям
    «Глобальная макроэкономика», «Российская макроэкономика»,
    «Российская региональная экономика», «Ключевые компании»:
    {news}

  4. **Наборы данных**
    • фактические — {data}
    • модельный прогноз — {forecast}
    Поля в наборах означают: «CPI» — инфляция в годовом выражении,
  «GDP» — реальный рост ВВП в годовом выражении,
  «ExchangeRate» — курс рубля к доллару,
  «KeyRate» — ключевая ставка в процентах,
  «InflationExpectations» — инфляционные ожидания населения в годовом выражении,
  «PriceExpectations» — инфляционные ожидания бизнеса в годовом выражении,
  «UralsPrice» — цена нефти марки «Urals», долларов за баррель,
  «Potrebaktivnost» — потребительская активность населения, темп роста к базовому периоду,
  «credit» — объем кредитования в млн руб.,
  «deposit» — объем депозитов в млн руб.,
  «ORT» - оборот розничной торговли в годовом выражении,
  «IPP» - индекс промышленного производства в годовом выражении.

  5. **Твои персональные характеристики** (будут переданы из python‑кода): {personality}

  ---

  ### Нечёткие («fuzzy») параметры сценария, которыми ты руководствуешься

  * «спрос» = {demand}
  * «предложение» = {supply}

  Они являются **непрерывными значениями** в диапазоне **[–2; 2]**:
  ‑2 — экстремально низкий уровень, 0 — нейтральный, +2 — экстремально высокий.
  Используй их как **веса принадлежности** при выборе и интерполяции между дискретными
  сценариями (из пункта 2). Чем ближе значение к конкретному целому уровню, тем выше
  степень принадлежности к соответствующему сценарию.
  В ответе **не упоминай** сами числа {demand} и {supply} и не раскрывай механику взвешивания.

  ---

  ## Задача

  **1.** Выведи прогноз конечных значений показателей на 31 декабря 2026 и 31 декабря 2027 гг.
  в формате JSON (строго без лишних полей):

  {{
    "2026": {{
      "CPI": ...,
      "GDP": ...,
      "KeyRate": ...,
      "ExchangeRate": ...,
      "UralsPrice": ...
    }},
    "2027": {{
      "CPI": ...,
      "GDP": ...,
      "KeyRate": ...,
      "ExchangeRate": ...,
      "UralsPrice": ...
    }}
  }}

  **2.** Подробно (250–300 слов) по‑русски обоснуй прогноз:

  * Чётко объясни, почему реализуется именно этот (смешанный) сценарий.
  * Опиши динамику и причины изменений по каждому показателю.
  * Основывайся на данных модели, реальной статистике и новостном фоне,
    а также на твоих персональных характеристиках.
  * Учитывай глобальные и российские факторы.
  * **Не** раскрывай внутренние веса, алгоритм интерполяции и свои персональные параметры.

  Верни **единую строку** (string) с JSON‑блоком и следом пояснительным текстом.
  """


def load_api_key(env_path=".env"):
    if os.path.exists(env_path):
        for line in Path(env_path).read_text().splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return os.getenv("API_KEY")

def load_agents(csv_path="agents.csv"):
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def get_agent_info(df, agent_id):
    cols = ["agent_id", "demand", "supply", "personality"]
    row = df.loc[df["agent_id"] == agent_id, cols]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def list_agents(df: pd.DataFrame) -> list:
    return df["agent_id"].tolist()


def read_answer(df: pd.DataFrame, agent_id: int) -> str:
    row = df[df["agent_id"] == agent_id]
    if row.empty:
        return ""
    return row.iloc[0]["answer"]


def call_judge(api_key, model_1, model_2, temperature=0.2):
    client = openai.OpenAI(api_key=api_key, base_url="https://bothub.chat/api/v2/openai/v1")
    prompt = PROMPT_TEMPLATE.format(model_1=model_1, model_2=model_2)
    print(prompt)
    response = client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        web_search_options={"search_context_size": "low"},
    )
    content = response.choices[0].message.content.strip()
    return content

def load_state_files(state_dir="state"):
    files = ["baseline.json", "data.json", "forecast.json", "news.json", "scenarios.json"]
    values = []
    for name in files:
        with open(os.path.join(state_dir, name), "r", encoding="utf-8") as f:
            values.append(f.read())
    return values  # baseline, data, forecast, news, scenarios


def call_model(api_key, demand, supply, personality, baseline, data, forecast, news, scenarios, temperature=0.6):
    prompt = GAUNTLET_PROMPT_TEMPLATE.format(
        demand=demand,
        supply=supply,
        personality=personality,
        baseline=baseline,
        data=data,
        forecast=forecast,
        news=news,
        scenarios=scenarios,
    )
    client = openai.OpenAI(api_key=api_key, base_url="https://bothub.chat/api/v2/openai/v1")
    response = client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    return response.choices[0].message.content


def thompson_sampling(stats):
    best_arm = None
    best_val = -1
    for arm, (a, b) in stats.items():
        val = random.betavariate(a, b)
        if val > best_val:
            best_val = val
            best_arm = arm
    return best_arm


def run_gauntlet(api_key, start_demand, start_supply, start_personality, rounds):
    baseline, data, forecast, news, scenarios = load_state_files()
    grid = [round(x, 1) for x in np.arange(-2, 2.01, 0.1)]
    personalities = [
        {"доверие_к_данным": d, "чувствительность_к_политическим_факторам": p, "стаж": s}
        for d in ["высокое", "умеренное", "низкое"]
        for p in ["низкая", "умеренная", "высокая"]
        for s in ["<5 лет", "5-10 лет", ">10 лет"]
    ]
    personalities_json = [
        json.dumps(p, ensure_ascii=False, sort_keys=True) for p in personalities
    ]
    arms = list(itertools.product(grid, grid, personalities_json))
    stats = {arm: [1, 1] for arm in arms}

    current_text = call_model(api_key, start_demand, start_supply, start_personality,
                              baseline, data, forecast, news, scenarios)
    start_personality_json = json.dumps(start_personality, ensure_ascii=False, sort_keys=True)
    current_arm = (round(start_demand, 1), round(start_supply, 1), start_personality_json)
    trajectory = [(current_arm[0], current_arm[1], current_arm[2])]
    yield {
        "round": 1,
        "leader": current_arm,
        "arm1": current_arm,
        "arm2": None,
        "text1": current_text,
        "text2": "",
        "trajectory": trajectory,
    }
    for r in range(2, rounds + 1):
        candidate_arm = thompson_sampling(stats)
        d, s, pers_json = candidate_arm
        pers_dict = json.loads(pers_json)
        cand_text = call_model(api_key, d, s, pers_dict,
                               baseline, data, forecast, news, scenarios)
        result = call_judge(api_key, current_text, cand_text)
        if result.strip() == "2":
            # candidate wins
            stats[candidate_arm][0] += 1
            stats[current_arm][1] += 1
            current_text = cand_text
            current_arm = candidate_arm
        else:
            stats[candidate_arm][1] += 1
            stats[current_arm][0] += 1
        trajectory.append((current_arm[0], current_arm[1], current_arm[2]))
        yield {
            "round": r,
            "leader": current_arm,
            "arm1": current_arm,
            "arm2": candidate_arm,
            "text1": current_text,
            "text2": cand_text,
            "trajectory": trajectory,
        }
    yield {
        "round": rounds,
        "leader": current_arm,
        "arm1": current_arm,
        "arm2": None,
        "text1": current_text,
        "text2": "",
        "trajectory": trajectory,
    }



def run_tournament(api_key, subset=None):
    agents_df = load_agents()
    competitors = list_agents(agents_df)
    if subset and subset > 1:
        competitors = random.sample(competitors, min(subset, len(competitors)))
    random.shuffle(competitors)

    round_num = 1
    result_log = ""
    while len(competitors) > 1:
        result_log += f"\n## Раунд {round_num}\n"
        winners = []
        for i in range(0, len(competitors), 2):
            if i + 1 >= len(competitors):
                winners.append(competitors[i])
                result_log += f"Агент {competitors[i]} проходит без боя.\n"
                continue
            a1 = competitors[i]
            a2 = competitors[i + 1]
            text1 = read_answer(agents_df, a1)
            text2 = read_answer(agents_df, a2)
            info1 = get_agent_info(agents_df, a1)
            info2 = get_agent_info(agents_df, a2)
            yield (
                str(info1),
                str(info2),
                text1,
                text2,
                "Сравнение...",
                result_log,
            )
            try:
                choice = call_judge(api_key, text1, text2)
            except Exception as e:
                choice = "error"
                result_log += f"Ошибка сравнения {a1} vs {a2}: {e}\n"
                yield (
                    None,
                    None,
                    None,
                    None,
                    "Ошибка сравнения",
                    result_log,
                )
                continue
            if choice.strip() == "1":
                winner = a1
            else:
                winner = a2
            winners.append(winner)
            result_log += f"Агент {winner} побеждает в паре {a1} vs {a2}.\n"
            yield (
                None,
                None,
                None,
                None,
                f"Победил агент {winner}",
                result_log,
            )
        competitors = winners
        round_num += 1
    champion = competitors[0]
    champion_info = get_agent_info(agents_df, champion)
    result_log += f"\n### Победитель турнира — агент {champion}""\n" + str(champion_info)
    yield (
        "",
        "",
        "",
        "",
        f"Победитель: агент {champion}",
        result_log,
    )


def start(api_key_text, subset=None):
    key = api_key_text or load_api_key()
    for update in run_tournament(key, subset=subset):
        yield update

def start_gauntlet(api_key_text, demand, supply, personality, rounds):
    key = api_key_text or load_api_key()
    personality_dict = json.loads(personality)
    for update in run_gauntlet(key, float(demand), float(supply), personality_dict, int(rounds)):
        yield (
            f"Раунд {update['round']}" ,
            json.dumps(update['leader'], ensure_ascii=False),
            json.dumps(update['arm1'], ensure_ascii=False),
            json.dumps(update['arm2'], ensure_ascii=False) if update['arm2'] else "",
            update['text1'],
            update['text2'],
            update['trajectory'],
        )


def build_interface(subset=None):
    with gr.Blocks() as demo:
        with gr.Tabs():
            # ---------- вкладка «Турнир» (без изменений) ----------
            with gr.TabItem("Турнир"):
                gr.Image(
                    "logo.png",
                    elem_id="logo",
                    width=240,
                    height="auto",
                    min_width=240,
                    show_label=False,
                    show_download_button=False,
                )
                api_key_inp = gr.Textbox(label="API Key", type="password", value=load_api_key() or "")
                start_btn = gr.Button("Начать соревнование")
                with gr.Row():
                    with gr.Column():
                        agent1_info = gr.Markdown()
                        text1 = gr.Textbox(lines=10, label="Ответ 1")
                    with gr.Column():
                        agent2_info = gr.Markdown()
                        text2 = gr.Textbox(lines=10, label="Ответ 2")
                result = gr.Markdown()
                log = gr.Markdown()

                def on_start(key):
                    yield from start(key, subset=subset)

                start_btn.click(
                    on_start,
                    inputs=api_key_inp,
                    outputs=[
                        agent1_info,
                        agent2_info,
                        text1,
                        text2,
                        result,
                        log,
                    ],
                )

            # ---------- вкладка «Гаунтлет» (обновлена) ----------
            with gr.TabItem("Гаунтлет"):
                gr.Image(
                    "logo.png",
                    elem_id="logo2",
                    width=240,
                    height="auto",
                    min_width=240,
                    show_label=False,
                    show_download_button=False,
                )
                api_key_g = gr.Textbox(label="API Key", type="password", value=load_api_key() or "")
                demand_inp = gr.Number(label="demand (-2..2)", value=0)
                supply_inp = gr.Number(label="supply (-2..2)", value=0)
                personality_inp = gr.Textbox(
                    label="personality JSON",
                    value='{"доверие_к_данным":"умеренное", '
                          '"чувствительность_к_политическим_факторам":"умеренная", '
                          '"стаж":"5-10 лет"}'
                )
                rounds_inp = gr.Number(label="Количество раундов", value=20)
                start_g_btn = gr.Button("Старт гаунтлета")

                # --- выводы ---
                round_info = gr.Markdown()
                leader_info = gr.Markdown()
                arm1_info = gr.Markdown()
                arm2_info = gr.Markdown()
                text1_box = gr.Textbox(lines=10, label="Модель 1")
                text2_box = gr.Textbox(lines=10, label="Модель 2")
                plot = gr.Plot(label="Траектория поиска")

                # --- Altair-чарт -------------------------------------------------
                import altair as alt

                def make_chart(df):
                    """
                    Серые точки — все предыдущие победители (подписаны round'ами);
                    Красный маркер — текущий лидер.
                    """
                    layers = []

                    if len(df) > 1:  # есть прошлые раунды
                        prev = df.iloc[:-1]

                        # точки-победители
                        layers.append(
                            alt.Chart(prev).mark_circle(
                                size=70,
                                color="lightgray"
                            ).encode(
                                x=alt.X("demand:Q", title="спрос"),
                                y=alt.Y("supply:Q", title="предложение"),
                            )
                        )

                        # подписи раундов
                        layers.append(
                            alt.Chart(prev).mark_text(
                                dx=7, dy=-7, fontSize=12, color="gray"
                            ).encode(
                                x="demand:Q",
                                y="supply:Q",
                                text="step:O"
                            )
                        )

                    # текущий победитель
                    current = df.iloc[[-1]]
                    layers.append(
                        alt.Chart(current).mark_circle(
                            size=250,
                            color="red",
                            stroke="black",
                            strokeWidth=2,
                        ).encode(
                            x="demand:Q",
                            y="supply:Q",
                        )
                    )

                    return alt.layer(*layers).properties(width=450, height=450)

                # --- функция запуска гаунтлета ----------------------------------
                def on_start_gauntlet(key, d, s, p, r):
                    for upd in start_gauntlet(key, d, s, p, r):
                        traj_df = pd.DataFrame(
                            [
                                {"demand": t[0], "supply": t[1], "step": idx + 1}
                                for idx, t in enumerate(upd[6])
                            ]
                        )
                        chart = make_chart(traj_df)

                        yield (
                            upd[0],  # round_info
                            upd[1],  # leader_info
                            upd[2],  # arm1_info
                            upd[3],  # arm2_info
                            upd[4],  # text1_box
                            upd[5],  # text2_box
                            chart,  # Altair-график
                        )

                # --- привязка кнопки -------------------------------------------
                start_g_btn.click(
                    on_start_gauntlet,
                    inputs=[api_key_g, demand_inp, supply_inp, personality_inp, rounds_inp],
                    outputs=[
                        round_info,
                        leader_info,
                        arm1_info,
                        arm2_info,
                        text1_box,
                        text2_box,
                        plot,
                    ],
                )
    return demo

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Randomly select this many agents for the tournament",
    )
    args = parser.parse_args()

    build_interface(subset=args.subset).launch()