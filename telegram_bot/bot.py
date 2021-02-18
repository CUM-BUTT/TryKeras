import telebot



bot = telebot.TeleBot('1604259440:AAHRI_3wrvj158k-oYVDxV47buwTQ1fYSZk')

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    print(message)

def Run():
    bot.polling(none_stop=True, interval=0)


if __name__ == '__main__':
    Run()