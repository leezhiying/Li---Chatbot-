import os 
os.chdir("/Users/lizhiying/Desktop/Big Data Analytics/LI-bot/chatbot_website")


from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify,make_response, abort
from word_predict import CornellWordChatBot





app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

cornell_word_chat_bot = CornellWordChatBot()
 

cornell_char_chat_bot_conversations = []
cornell_word_chat_bot_conversations = []
cornell_word_glove_chat_bot_conversations = []



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return 'About Us'


@app.route('/char_reply', methods=['POST', 'GET'])
def cornell_char_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            cornell_char_chat_bot_conversations.append('YOU: ' + sent)
            #reply = cornell_char_chat_bot.reply(sent)
            cornell_char_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('cornell_char_reply.html', conversations=cornell_char_chat_bot_conversations)


@app.route('/word_reply', methods=['POST', 'GET'])
def cornell_word_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            cornell_word_chat_bot_conversations.append('YOU: ' + sent)
            reply = cornell_word_chat_bot.reply(sent)
            cornell_word_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('word_reply.html', conversations=cornell_word_chat_bot_conversations)



@app.route('/chatbot_reply', methods=['POST', 'GET'])
def chatbot_reply():
    if request.method == 'POST':
        if not request.json or 'sentence' not in request.json or 'level' not in request.json or 'dialogs' not in request.json:
            abort(400)
        sentence = request.json['sentence']
        level = request.json['level']
        dialogs = request.json['dialogs']
    else:
        sentence = request.args.get('sentence')
        level = request.args.get('level')
        dialogs = request.args.get('dialogs')

    target_text = sentence
    
    if level == 'char' and dialogs == 'cornell':
        target_text = cornell_char_chat_bot.reply(sentence)
    
    if level == 'word' and dialogs == 'cornell':
        target_text = cornell_word_chat_bot.reply(sentence)

    return jsonify({
        'sentence': sentence,
        'reply': target_text,
        'dialogs': dialogs,
        'level': level
    })


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def main():
    cornell_word_chat_bot.test_run()
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    main()
