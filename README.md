# ai_in_ir

Веб-сервис, который по базе знаний умеет отвечать на вопросы пользователя. Для успешной работы необходимо положить в папку apps файл llm-модели.

команда "docker build −t llm_username : v1 ." собирает контейнер

команда "docker run −p 8080:8080 llm_username : v1" запускает контейнер

Пример запроса: - curl -X POST -H "Content-Type: application/json" -d '{"message": "Сколько стоят смски с оповещениями об операциях", "user_id": "1232"}' http://localhost:8080/message
