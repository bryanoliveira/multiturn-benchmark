def generate_chat(
    user_pipeline,
    assistant_pipeline,
    user_system_prompt,
    assistant_system_prompt="You are a helpful assistant. Your responses are always concise and to the point.",
    first_user_message="Help me write an email.",
    max_turns=5,
    debug=False,
    max_new_tokens=500,
):
    user_history = [
        {
            "role": "system",
            "content": user_system_prompt,
        },
    ]
    assistant_history = [
        {
            "role": "system",
            "content": assistant_system_prompt,
        },
        {
            "role": "user",
            "content": first_user_message,
        },
    ]

    if debug:
        print("USER SYSTEM: ", user_history[0]["content"])
        print("ASSISTANT SYSTEM: ", assistant_history[0]["content"])
        print("-" * 10)

    for i in range(max_turns):
        # user sent a message to the assistant
        assistant_response = assistant_pipeline(
            assistant_history,
            max_new_tokens=max_new_tokens,
            pad_token_id=assistant_pipeline.tokenizer.eos_token_id,
        )[0]["generated_text"][-1]["content"]

        if debug:
            print(f"\n\n{i+1} - ASSISTANT: {assistant_response}")

        assistant_history += [{"role": "assistant", "content": assistant_response}]
        user_history += [{"role": "user", "content": assistant_response}]

        # assistant sent a message to the user
        user_response = user_pipeline(
            user_history,
            max_new_tokens=max_new_tokens,
            pad_token_id=user_pipeline.tokenizer.eos_token_id,
        )[0]["generated_text"][-1]["content"]

        if debug:
            print(f"{i+1} - USER: {user_response}")

        assistant_history.append({"role": "user", "content": user_response})
        user_history.append({"role": "assistant", "content": user_response})

    # user sent a message to the assistant
    assistant_response = assistant_pipeline(
        assistant_history,
        max_new_tokens=max_new_tokens,
        pad_token_id=assistant_pipeline.tokenizer.eos_token_id,
    )[0]["generated_text"][-1]["content"]

    if debug:
        print(f"{i+1} - ASSISTANT: {assistant_response}")

    return user_history, assistant_history
