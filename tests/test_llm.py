from app.llm import merge_system_into_user


def test_system_is_prepended_to_first_user_message():
    messages = [
        {"role": "system", "content": "Правила"},
        {"role": "user", "content": "Вопрос"},
        {"role": "assistant", "content": "Ответ"},
        {"role": "user", "content": "Ещё вопрос"},
    ]
    merged = merge_system_into_user(messages)
    assert [m["role"] for m in merged] == ["user", "assistant", "user"]
    assert merged[0]["content"] == "Правила\n\nВопрос"
    assert merged[2]["content"] == "Ещё вопрос"
    assert messages[1]["content"] == "Вопрос"  # input is not mutated


def test_without_system_messages_are_unchanged():
    messages = [{"role": "user", "content": "Вопрос"}]
    assert merge_system_into_user(messages) == messages


def test_system_only_becomes_user():
    assert merge_system_into_user([{"role": "system", "content": "Правила"}]) == [
        {"role": "user", "content": "Правила"}
    ]
