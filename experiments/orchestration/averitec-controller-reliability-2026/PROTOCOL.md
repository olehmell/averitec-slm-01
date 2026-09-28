# Протокол: виконання інструкцій у багатокроковому контролері

Версія: `controller-reliability-v4-fixed-evaluation-100`, 2026-09-19.
Користувач погодив 100-case evaluation та scoped local commit без push.
Lifecycle `running`; профілі нижче зафіксовані до нової qualification/evaluation.

## Поточний погоджений запуск: 100 нових cases

Цей розділ визначає поточний обмежений запуск; screening-розділи нижче
зберігають історію вибору налаштувань, але не розширюють його матрицю.

П'ять фіксованих умов: Jev v2/baseline, Qwen v2/baseline,
Gemini v2/gemini_native, LFM1.2 v3/lfm_native, LFM26 v3/lfm26_native.
Вибір зроблено за завершеним двокейсовим screening і погоджено користувачем.
Моделі, revisions, prompts, output limits та state-machine rules не змінюються.
Це порівняння попередньо обраних deployments, не чистий common-prompt contrast.

1. Qualification: наступні чотири development groups, canonical slice `[2:6]`,
   які не входили у двокейсовий screening. Три сценарії; canonical тричі,
   reversed і seeded order по одному разу. П'ять моделей: 300 trials,
   максимум 3,600 measured decisions і п'ять окремих warmups.
2. Перед evaluation перевіряються всі п'ять complete runs, matched case/order
   plans і freeze, hashes, successful warmups, відсутність interrupted trials
   та transport/version failures. Repeatability та order sensitivity
   обчислюються й зберігаються. Низька compliance, valid wrong actions або
   schema/truncation failures є результатами, не приводом відсіяти модель чи
   змінити prompt. Немає довільного accuracy pass threshold.
3. Після валідного portable qualification receipt — новий shared Qwen freeze
   усіх 100 канонічних evaluation cases. Попередні development cases сюди
   не входять; gold і main-study holdout залишаються недоступними.
4. Evaluation: checkpoint mode, один canonical pass, три сценарії,
   100 cases × 3 × 5 = 1,500 trials, максимум 18,000 measured decisions
   плюс п'ять warmups. Жодного post-result tuning, автоматичного repair,
   semantic retry, заміни кейсів або відбору лише successful preparations.

Фіксація профілів сама по собі не відкриває evaluation: потрібен receipt,
що перевіряє qualification evidence під тими самими code/config bindings.
Проста зміна `evaluation_settings_frozen` не замінює цей gate.
Якщо qualification некомплектна або є infrastructure/identity failure,
evaluation не починається; причина й часткові результати зберігаються.

Наявні images/caches використовуються повторно. Жорстка сума планових
GPU time limits — 12 allocation-hours (не прогноз витрат): qualification
freeze 1 h, LFM/Qwen по 15 min, LFM26 30 min; evaluation freeze 6 h,
LFM/Qwen по 30 min, LFM26 3 h. API phase на модель — до 4 h із request
caps вище. Реальний час може бути значно меншим; черга не входить у GPU
allocation-hours. Вичерпання ліміту або невідомий збій — stop/report,
не мовчазний повтор запитів. Підготовка shared-worker обліковується окремо
від measured controller latency та має незмінні per-case caps.

Критерій завершення: 100 canonical cases у новому freeze, п'ять complete
evaluation journals із 300 trials кожен, matched checkpoints, перевірені
result/intent/trace/warmup receipts, resource accounting і підсумковий звіт.
Trajectory completion, evaluation repeatability та evaluation order sensitivity
цей single-pass запуск не оцінює; останні дві перевіряються лише на qualification.

## Дослідницьке питання

За однакових спостережень і правил процесу, наскільки стабільно Jev, дві LFM,
Qwen та Gemini обирають нормативно правильний наступний крок і завершують процес?
Це операційне визначення виконання інструкцій; воно не охоплює всі види
reasoning або інструкцій природною мовою. Якість verdict/evidence не оцінюється.

Основна гіпотеза двостороння: частка first-attempt compliant decisions
відрізняється між розгорнутими controller systems. Напрям переваги
заздалегідь не стверджується. Форматна коректність не дорівнює виконанню правил.

## Arms і спільні компоненти

| Компонент | Binding |
| --- | --- |
| Jev controller | `jev-1.13.0`, TypeSafe HTTP Choice, без retries |
| LFM controller | `LiquidAI/LFM2.5-1.2B-Instruct`, pinned revision із config |
| LFM26 controller | `LiquidAI/LFM2.5-2.6B`, окремий always-thinking deployment, pinned revision із config |
| Qwen controller | `Qwen/Qwen3.5-4B`, pinned revision із config; thinking off |
| Gemini controller | `gemini-3.1-flash-lite`, native GenerateContent JSON; явний minimal thinking |
| Text tools та verdict | Один Qwen3.5-4B з однаковими frozen prompts/params |
| Retrieval | CPU BM25, однаковий per-claim corpus і answer-page exclusions |
| Executor | Детермінована state machine, окремий validator та журнал |

Jev повертає typed decisions замість довільного тексту
([офіційний опис](https://docs.typesafe.ai/introduction)); тому генерація
queries/QA/verdict винесена у спільний worker. Його metrics-only projection
містить етап, completed stages, retry/call counters, outcome enum та counts.
Claim, QA, джерела, URLs, ID кейса, label і evaluator feedback контролеру недоступні.
Це новий observation contract; старий контракт основного експерименту не змінено.

Порівняння характеризує deployments із різними моделями та інтерфейсами.
Воно не ізолює причинний ефект архітектури, числа параметрів або тренувальних даних.

## Погоджене розширення: Gemini, LFM 2.6B і компактний протокол

Зберігаємо обидві LFM як окремі arms: `lfm` та `lfm26`. Нова команда
`--expansion-screen` використовує тільки матрицю `expansion_screening` із config:

| Controller | Prompt variants | Generation profile | Умов |
| --- | --- | --- | ---: |
| Jev | v2, v3 | baseline | 2 |
| Qwen | v2, v3 | baseline, thinking off | 2 |
| LFM 1.2B | v2, v3, v4 | lfm_native | 3 |
| LFM 2.6B | v2, v3, v4 | lfm26_native, always thinking | 3 |
| Gemini Flash-Lite | v2, v3 | gemini_native, explicit minimal thinking | 2 |

Рівно два development cases, три вже визначені scenarios, один canonical
pass: **12 умов, до 864 виміряних calls і 12 окремих warmup calls**. Actual
checkpoint count визначає новий freeze, тому попередні 46 рішень на умову
не припускаються автоматично. Це не tuning на 100 evaluation cases.

`v4` змінює лише інструкції: компактний пріоритет правил, з явним
`unfinished = 7 - count(completed_tools)`. Presentation лишається v2,
усі metrics і всі дев'ять дій збережені; прикладів немає. Різниця v2/v4
досліджує компактність інструкцій без вилучення інформації зі стану.

`lfm26_native`: temperature 0.1, top_k 50, repetition_penalty 1.1,
2048 output tokens, timeout 60 s. Режим reasoning невимиканий у
[моделі](https://huggingface.co/LiquidAI/LFM2.5-2.6B/tree/654f9463ce32b05d0429d76fe1f580b27d4c1ac0).
Старий 160-token baseline для неї відхиляється. Кандидат runtime — vLLM
0.28.0/Transformers >=5.5.3; попередній vLLM 0.17.0 образ несумісний із
її TokenizersBackend. Збережено окремий pinned image recipe; до вимірювань
потрібні offline config/tokenizer check та успішний warmup із непорожнім
окремим reasoning, валідним final JSON і stop. `qwen3` parser тут є
перевірюваною технічною гіпотезою, не заявою офіційної LFM-сумісності.

`gemini_native`: native REST v1beta, temperature 1, 2048 maxOutputTokens,
explicit `thinkingLevel: minimal`; мінімальний не означає guaranteed zero
thinking. Немає grounding/tools, automatic retries або text repair. Ключ
читається тільки з `GEMINI_API_KEY` і передається в заголовку; у журналах
й receipts ключа/заголовка немає. Невалідний JSON, safety block, порожня чи
обрізана відповідь — failures. Output usage включає candidate + thought
tokens, без додавання totalTokenCount вдруге; відсутні required counts — unknown.

GET model metadata підтвердив доступ і `version: 3.1-flash-lite-05-2026`.
Це availability metadata, а не deployment revision pin. Контрольована одна
synthetic qualification на commit `00bf272` спостерегла
`GenerateContent.modelVersion: gemini-3.1-flash-lite`; саме цей alias є
strict expected identity у config. Mismatch зупиняє run; alias/version не
переписуються автоматично. Metadata version зберігається окремо, бо Google не
гарантує однакового написання двох полів. Qualification receipt і trace лишаються
private; його один synthetic call не є measured decision.
Джерела: [модель](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite),
[GenerateContent](https://ai.google.dev/api/generate-content),
[thinking](https://ai.google.dev/gemini-api/docs/thinking).

Зміни code/config вимагають нового real-tool freeze під поточними bindings.
Старі worker records та hash bindings не переписуються; controls Jev/Qwen
виконуються заново на тих самих нових спостереженнях, що й нові arms.
Історичні результати з іншого freeze не об'єднуються в paired estimate.
Навіть спільний prompt не ізолює число параметрів: моделі також відрізняються
навчанням, reasoning і runtime. Profiles порівнюються як deployments.
Summary зберігає profile labels; cross-model contrasts автоматично доступні
лише для однакових profile labels, native arms порівнюються явно описово.

## Історичний screening: зрозуміліший протокол трьох моделей

Користувач погодив 2026-09-19 перевірити ясніший протокол, навчальні приклади
та сильніші доступні режими кожної з трьох pinned моделей. Попередній dev2
smoke — діагностика, не фінальний рейтинг і не підстава вибору за evaluation.
Старі результати не перезаписуються. Усі умови нижче виконуються на одному
новому freeze, який зв'язує актуальний код; старий freeze не переприв'язується.

Два окремі питання:

1. **Спільний протокол:** v1 (точні старі інструкції/стан), v2 (явний pending
   tool, названий попередній результат, пріоритетний алгоритм), v3 (v2 плюс
   шість синтетичних прикладів). Усі три моделі проходять усі варіанти з
   `baseline` generation profile. Це порівняння представлення стану та правил.
2. **Можливості deployment:** для v2 і v3 окремо додаються native profiles
   із таблиці. Профіль — пакет налаштувань, а не ізольований причинний фактор.
   Його не об'єднують із baseline або умовою іншої моделі під спільною назвою.

| Profile | Зміна проти baseline |
| --- | --- |
| `jev_native` | Статичні string criteria для всіх дев'яти Choice options |
| `lfm_native` | temperature 0.1, top_k 50, repetition_penalty 1.05; 160 tokens |
| `qwen_native` | thinking off; temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, repetition_penalty 1; 160 tokens |
| `qwen_thinking` | thinking on; temperature 1, top_p 0.95, top_k 20, presence_penalty 1.5, repetition_penalty 1; 2048 tokens, timeout 60 s |

Параметри походять із карт саме [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
і [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct),
а string criteria — з [TypeSafe HTTP API](https://docs.typesafe.ai/api).
Рекомендація авторів не гарантує кращий результат для нашого завдання.
Для Qwen server використовує `qwen3` reasoning parser; structured constraint
застосовується до фінальної відповіді, не до reasoning. Raw reasoning, final
content, finish reason і usage зберігаються у private traces; обрізана відповідь
є помилкою, а не приводом повторити запит з більшим бюджетом.

v2/v3 не додають фактичних текстів, gold або oracle response; counts збережені.
Вони лише перейменовують уже наявні поля й явно зв'язують last status із tool.
Приклади написані з нормативних правил, не вибрані з evaluation. Native Jev
criteria статичні: vocabulary не залежить від state. Немає correct-action mask,
парсингу з автоматичним ремонтом, голосування або повторного запиту після помилки.

Початковий screening: перші два development cases, усі три scenarios, canonical
order, один repetition. Це 5 умов Jev, 5 LFM і 7 Qwen; не більше 1,224 measured
decision calls (17 умов × 2 cases × 3 scenarios × cap 12), плюс 17 warmup calls.
Кожний локальний controller — один Vega job із наявним лімітом 30 хвилин;
підготовка нового спільного freeze — окремий bounded job. Без нових моделей,
навчання, оренди ресурсів або автоматичного розширення пошуку.

Перед кожною умовою — рівно одна synthetic генерація тим самим controller
adapter/schema з окремим timeout 120 s. Успіх означає валідний формат та exact
model identity, не правильну за oracle дію. Warmup journals/usage відокремлені
від вимірювань; збій зупиняє запуск умови. GET `/v1/models` сам по собі не є
підтвердженням готовності генерації. Initial cold start обліковується окремо.

Після screening можна рекомендувати кандидатів, але не оголошувати глобальний
«максимум» чи переможця. Перед evaluation потрібні повтори/option-order checks
і перевірка на інших development groups, однаковий tuning budget та явна фіксація
обраних налаштувань. `evaluation_settings_frozen: false` блокує tuned evaluation
в CLI. Усі 100 evaluation cases залишаються закритими для підбору.

Summary розділяє prompt variant та generation profile, показує transport/schema
failures, rule compliance і витрати. Paired configuration contrasts — лише
описові; численні development comparisons не є confirmatory significance tests.

## Нормативні правила і всі етапи

`decompose → queries → retrieve → qa → coverage → select → verdict → finish`.
Робота з текстом реальна при створенні freeze та в optional live-tool smoke.
Frozen trajectory відтворює результати тих самих tools без повторної генерації.
Завершення означає проходження всіх семи tools і явний `finish`.

- `ok` і `empty` завершують поточний етап, після них потрібен наступний.
- `timeout`/`invalid` не завершують етап: одна повторна спроба тієї самої дії.
- Друга невдала спроба вимагає `abort`.
- Якщо remaining calls менше за мінімум незавершених tools + finish, потрібен abort.
- Передчасний finish, безпідставний abort, пропуск або повтор завершеного
  етапу є порушеннями. Validator не виконує таку дію і не виправляє відповідь.

Очікувана дія виводиться з цих правил до model call; це не gold factual label
і не суб'єктивна «найкраща» стратегія. Deterministic oracle перевіряє harness.
Без нього повторення моделі й збої state machine було б важко відокремити.

У всіх primary model requests доступні всі дев'ять actions, включно з
неправильними на поточному етапі. Jev Choice і LFM/Qwen JSON-schema enum
мають однакову область відповідей. Current-stage mask не використовується:
інакше type constraint механічно усував би помилки, які ми вимірюємо.
Prompt-only LFM/Qwen — окрема абляція формату, поза primary arm grouping.

## Дані та контроль витоку

Вибірка: original-dev records усередині наявного `study_fit`, один кейс на
transitive duplicate group; seed `20260918`, 20 development і 100 evaluation.
Так достатньо одного dev knowledge store; train shards не блокують експеримент.
Власний selection manifest відділяє новий evaluation від development. Main
`study_holdout` та `averitec-dev-0002` попереднього Jev probe виключено.
Поточний CLI заново перевіряє детермінований selection; довільна підміна IDs
або stale input hashes відхиляються.

Gold files не читаються. Existing split metadata використовується тільки для
group/role membership; worker бачить claim і дозволений corpus. Аналітичні
висновки стосуються інструкцій, а не unseen fact-check knowledge. Попередня
експозиція AVeriTeC моделі чи досліднику не оголошується усунутою новим split.

У старому чотиривикличному probe кількості були сконструйовані за gold QA.
Він залишається перевіркою API і не є даними цього експерименту. Нові
спостереження походять від фактичних shared-tool outputs із receipts.

## Два основні режими й operational smoke

1. **Checkpoints (primary).** На кожному frozen state кожна модель незалежно
   обирає дію. Помилка на ранньому кроці не прибирає пізні checkpoints з тесту.
2. **Trajectory (secondary).** Контролер проходить ті самі frozen tool results.
   Перше порушення завершує trial; втручання executor не рахується успіхом.
3. **Live tools (development smoke).** Контролери керують повторним фактичним
   виконанням shared tools. Цей режим перевіряє інтеграцію; worker variability
   і його costs обліковуються окремо. Його не змішують із frozen replay.

Сценарії: normal; один retrieval timeout перед dispatch; два послідовні
retrieval timeouts. Останній сценарій має нормативний abort: це правильне
завершення, але не full-pipeline completion. Це контрольовані fault injections,
не спостережена частота відмов реального пошуку.

Canonical option order повторюється тричі з однаковим seed/temperature=0;
reversed і seed-permuted orders — по одному разу, зіставлення з canonical
repeat 0. Це розділяє repeatability і sensitivity до представлення варіантів.
Jev version pinned; actual returned version записується. Confidence Jev —
характеристика distribution, не незалежна оцінка правильності
([Choice](https://docs.typesafe.ai/primitives/choice)).

Для 100 успішно підготовлених cases номінальний план primary становить
100 × (8 + 9 + 5) states × 5 repetition/order conditions × 3 models = 33,000
decisions. Це оцінка обсягу, не вже виконана робота й не power calculation.
Початковий smoke — 2 cases × 8 nominal states на модель. Реальні costs,
контекст, throughput і потрібний розмір evaluation перевіряються на development.

## Метрики й статистичний план

Primary: compliant decisions / усі заплановані checkpoint decisions.
Невиміряні через interruption відповіді не вважаються успіхом. Окремо показуються
compliance conditional on valid response, malformed outputs і transport failures.
Це дозволяє відрізнити недотримання правила від недоступності сервісу.

Secondary: autonomous full completion, protocol-correct termination, early
finish, retry violations, repeatability, option-order sensitivity, median/p95
controller wall latency, measured tokens та missing usage. Token sums між
різними tokenizer-ами не є нормалізованою compute metric. Worker preparation
costs не входять у controller replay latency; Jev network time входить.

Paired contrasts обчислюються на тих самих case/scenario observations.
Повтори агрегуються всередині case/group; 95% bootstrap intervals використовують
ці групи, не кожен call як незалежний sample. Три contrasts описові; окремого
непоправленого significance claim не робити. Формальний confirmatory test або
power claim потребує додаткового statistical freeze після development pilot.

Різні cases можуть мати ідентичні числові observations. Звітувати кількість
унікальних observation hashes; sample size не прирівнювати до кількості API
викликів. Ідентичні стани не створюють нових незалежних instruction tasks.

## Відтворюваність і запуск

Freeze зв'язує code bytes, config, selection/runtime/split hashes, source
registry/exclusions, pinned worker та per-stage receipts із model-call counts
і content hashes. Raw/private тексти лишаються поза tracked outputs.
Controller records містять normalized response, request-state hash, actual
model identity, латентність і usage; жодних secrets.

Обов'язкові локальні trace journals зберігають start/end timestamps, parent-child
зв'язки, controller prompts та відповіді, дозволені worker prompts/completions і
стани tools. Повний retrieval ranking у trace зв'язується кількістю та SHA-256
впорядкованих passage IDs; текст зберігається лише для точного 10-passage QA
window, а не дублює весь per-claim corpus. Окрема межа у 2 MB на одну trace-подію
fail-closed захищає локальне сховище. Кожен trial посилається на trace попередньої
shared-tool preparation.
Текстові артефакти не є controller observations і не читають gold. Експорт у
Langfuse відкладений до завершення вимірювань; remote availability не є умовою
виконання експерименту. Provider latency вимірюється окремо від trace I/O;
повний span duration може містити локальні накладні витрати. Журнали з текстами
лишаються ignored/private, публікуються лише перевірені редаговані приклади.

Є журнал intent-before-call, run lock і перевірка digest результатів. Unknown
outcome після interruption не запускається автоматично знову. Всі planned
trials залишаються у denominator; worker preparation failures видно окремо.
Неповний summary не є фінальним результатом paper: перевірити planned/observed
cardinality для всіх трьох arms і однаковий freeze.

LFM і Qwen запускаються на Vega послідовно на одному GPU; exact account і
allocation units перевіряються перед job. Jev replay може виконуватися локально
за тим самим freeze: це знімає залежність від зовнішнього API з compute nodes.
Latency такої схеми характеризує deployments, не чисту GPU inference speed.
Власні output directories дозволяють працювати паралельно з основним дослідженням.

## Межі висновку й стаття

Можна оцінити дотримання заданого workflow contract і виділити характерні
помилки. Не можна з цих результатів вивести factual accuracy, general agent
intelligence, перевагу topology або покращення основного експерименту.
Повністю deterministic routing є корисним engineering baseline; цей protocol
навмисно досліджує, чи виконують такі правила самі моделі. До deployment варто
окремо обґрунтувати, навіщо в подібній state machine потрібен learned controller.

Фактичні результати та готовність до статті відображає
[evidence map](publication/evidence-map.md). [Модельна документація Jev](https://docs.typesafe.ai/models)
підтверджує version pinning і поточний API; вона не підтверджує наші empirical claims.
