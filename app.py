import streamlit as st
import torch
from torch import nn
from torchtext.data.utils import get_tokenizer

class LanguageModel(nn.Module):

  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rnn=0.5, dropout_embd=0.5):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, embedding_dim)
    self.emb.weight.data.uniform_(-0.1, 0.1)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_rnn, batch_first=True)
    self.fc = nn.Linear(hidden_dim, vocab_size)
    self.dropout = nn.Dropout(dropout_embd)
  def forward(self, src):
    embedding = self.dropout(self.emb(src))
    output, _ = self.lstm(embedding)
    prediction = self.fc(output)
    return prediction

embedding_dim = 300

num_layers = 3
hidden_dim = 1150
dropoute = 0.1
dropouti = 0.65
dropouth = 0.3
dropouto = 0.4
weight_drop = 0.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('model.pt', map_location=torch.device(device))
model.eval()
tokenizer = get_tokenizer('basic_english')
vocab = torch.load('vocab.pt')

LanguageModel(len(vocab), 300, 512, 2)

def generate(prompt, tokenizer=tokenizer, vocab=vocab, model=model, max_seq_len=6, temperature=0.5, num_pred=4, seed=None):
  if seed is not None:
    torch.manual_seed(seed)


  itos = vocab.get_itos()
  preds = []
  for _ in range(num_pred):
    seq = prompt
    indices = vocab(tokenizer(seq))
    itos = vocab.get_itos()
    for i in range(max_seq_len):
      src = torch.LongTensor(indices).to(device)
      with torch.no_grad():
        prediction = model(src)

      probs = torch.softmax(prediction[-1]/temperature, dim=0)

      idx = vocab['<ukn>']
      while idx == vocab['<ukn>']:
        idx = torch.multinomial(probs, num_samples=1).item()

      token = itos[idx]
      seq += ' ' + token

      if idx == vocab['.']:
        break

      indices.append(idx)
    preds.append(seq)

  return preds


background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://manybackgrounds.com/images/hd/google-sunset-mountains-chrome-theme-fptnw9bzb312e05l.webp");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

st.markdown(
    """
    <style>

    .search-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 40vh;
        flex-direction: column;
        width: 60%;
    }
        .suggestion {
        background-color: #eaeaea;
        padding: 4px;
        border-radius: 10px; /* گوشه‌ها به اندازه 15px گرد شده‌اند */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        position: relative;
        top: 50%;
        left: 12px;
        font-size: 16px;
        color: #777;
        pointer-events: none;
        width: 95%;
    }
    .search-bar {
        width: 50%;
        position: relative;
    }
    .user-input {
        width: 100%;
        padding: 10px;
        font-size: 20px;
        border: 2px solid #ddd;
        border-radius: 25px;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.2);
        color: #000;
    }

    .suggestions {
        margin-top: 20px;
        
        font-size: 18px;
        color: #333;
        text-align: left;
        width: 50%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ذخیره وضعیت ورودی کاربر
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""

# ساختار HTML برای نوار جستجو و جملات پیشنهادی
st.markdown('<div class="search-container">', unsafe_allow_html=True)

# نوار جستجو بدون دکمه ارسال
user_input = st.text_input("", placeholder="Enter a word or phrase...", label_visibility="collapsed")

# به‌روزرسانی پیشنهادات هنگام تغییر ورودی کاربر
if user_input != st.session_state['user_input']:
    st.session_state['user_input'] = user_input

# فقط اگر ورودی کاربر خالی نباشد، پیشنهادات نمایش داده می‌شوند
if st.session_state['user_input']:
    suggestions = generate(st.session_state['user_input'])


    for suggestion in suggestions:
        st.markdown(
        f"""
        </div>
            <div class="suggestion">{suggestion}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
else:
    pass

st.markdown('</div>', unsafe_allow_html=True)
