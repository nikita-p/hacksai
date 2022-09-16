# Чемпионат

[Прогнозирование статуса студента](https://hacks-ai.ru/championships/758263)

Решение лежит в файле `notebook.ipynb`

* preprocessing v3 (0.779768): 
  * Здесь я добавил поле `МестоУчебы`
* preprocessing v4 (0.796647): 
  * Изменил поле `МестоУчебы` на `МестоЖит`, поправил его
  * Удалил малоинформативные поля типа `Общежитие`, `Изучал_Англ`
* catboost weighted classes 1.8 (0.797044):
  * Подкрутил веса классам при обучении catboost