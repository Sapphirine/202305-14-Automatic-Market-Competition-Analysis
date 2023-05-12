/** 获取表单数据 */
function getFormData() {
  const category = document.getElementById("category").value;
  const tag = document.getElementById("tag").value;
  const genre = document.getElementById("genre").value;
  return { category, tag, genre };
}

const BASE_URL = "http://localhost:5000/";
function hide_all_block() {
  document.getElementById("user_input").style.display = "none";
  document.getElementById("stage").style.display = "none";
  document.getElementById("similar_games").style.display = "none";
  document.getElementById("game_stats_analysis").style.display = "none";
  document.getElementById("pos_user_reviews").style.display = "none";
  document.getElementById("neg_user_reviews").style.display = "none";
}
async function submitForm() {
  // e.preventDefault();
  hide_all_block();
  const formData = getFormData();

  document.getElementById(
    "user_input"
  ).textContent = `Category: ${formData.category}  |
           Tag: ${formData.tag}       |
           Genre: ${formData.genre} `;
  document.getElementById("user_input").style.display = "block";
  document.getElementById("stage").textContent = "Calculating...";
  document.getElementById("stage").style.display = "block";
  const res = await fetch(BASE_URL + "submit", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: `category=${formData.category}&tag=${formData.tag}&genre=${formData.genre}`,
    mode: "cors",
  });
  let text = await res.text();
  document.getElementById("stage").textContent = text;
  if (text === "Success") {
    process_data();
  }
}

async function process_data() {
  document.getElementById("stage").textContent = "Processing Data...";
  const res = await fetch(BASE_URL + "data_process", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: ``,
    mode: "cors",
  });
  let text = await res.text();
  document.getElementById("stage").textContent = text;
  train_lda();
}

async function train_lda() {
  document.getElementById("stage").textContent = "Training LDA...";
  const res = await fetch(BASE_URL + "train_lda", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: ``,
    mode: "cors",
  });
  let text = await res.text();
  document.getElementById("stage").textContent = text;
  similar_docs();
}

async function similar_docs() {
  document.getElementById("stage").textContent = "Finding Similar Docs...";
  const res = await fetch(BASE_URL + "similar_docs", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: ``,
    mode: "cors",
  });
  let data = await res.json();

  let html = "<h2 style='text-align: center;'>Similar Games</h2>";
  for (let i = 0; i < data.game_names.length; i++) {
    html += `<h3>${data.game_names[i]}</h3><p>${data.game_desc[i]}</p>`;
  }
  document.getElementById("similar_games").innerHTML = html;
  document.getElementById("similar_games").style.display = "block";

  game_stats();
}

async function game_stats() {
  document.getElementById("stage").textContent = "Game Stats Analysis...";
  const res = await fetch(BASE_URL + "game_stats", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: ``,
    mode: "cors",
  });
  let paths = await res.json();

  let html = "<h2 style='text-align: center;'>Game Stats Analysis</h2>";
  for (let i = 0; i < paths.length; i++) {
    html += `<img src="${paths[i]}" style="width:50%">`;
  }
  document.getElementById("game_stats_analysis").innerHTML = html;
  document.getElementById("game_stats_analysis").style.display = "block";
  pos_user_reviews();
  neg_user_reviews();
}

async function pos_user_reviews() {
  document.getElementById("stage").textContent = "User Review Analysis...";
  const res = await fetch(BASE_URL + "pos_user_reviews", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: ``,
    mode: "cors",
  });
  let summary = await res.json();
  let words = summary.cluster_summary;

  let html = "<h2 style='text-align: center;'>Positive Reviews</h2>";
  for (let word of words) {
    html += `<span>${word}</span>`;
  }

  let paths = summary.word_cloud;
  for (let path of paths) {
    html += `<img src="${path}" style="width:50%">`;
  }

  document.getElementById("pos_user_reviews").innerHTML = html;
  document.getElementById("pos_user_reviews").style.display = "block";
  document.getElementById("stage").style.display = "none";
}

async function neg_user_reviews() {
  document.getElementById("stage").textContent = "User Review Analysis...";
  const res = await fetch(BASE_URL + "neg_user_reviews", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: ``,
    mode: "cors",
  });
  let summary = await res.json();
  let words = summary.cluster_summary;

  let html = "<h2 style='text-align: center;'>Negative Reviews</h2>";
  for (let word of words) {
    html += `<span>${word}</span>`;
  }

  let paths = summary.word_cloud;
  for (let path of paths) {
    html += `<img src="${path}" style="width:50%">`;
  }

  document.getElementById("neg_user_reviews").innerHTML = html;
  document.getElementById("neg_user_reviews").style.display = "block";
}

const submitButton = document.getElementById("sendDataBtn");
submitButton.addEventListener("click", submitForm);
