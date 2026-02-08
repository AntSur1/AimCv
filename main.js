const POSTS_DIR = "posts";
const entriesEl = document.getElementById("entries");

/**
 * You must list files manually or auto-generate this list via a build step.
 */
const postFiles = [
  "2026-02-08-previously.txt",
];

async function loadPost(filename) {
  const res = await fetch(`./${POSTS_DIR}/${filename}`);
  const text = await res.text();

  const [, frontmatter, content] =
    text.match(/---([\s\S]*?)---([\s\S]*)/) || [];

  const meta = Object.fromEntries(
    frontmatter
      .trim()
      .split("\n")
      .map(line => line.split(":").map(s => s.trim()))
  );

  return {
    title: meta.title || filename,
    date: meta.date || "",
    html: marked.parse(content)
  };
}

async function loadAll() {
  const posts = await Promise.all(postFiles.reverse().map(loadPost));

  // newest first
  posts.sort((a, b) => new Date(b.date) - new Date(a.date));

  for (const post of posts) {
    const article = document.createElement("article");
    article.innerHTML = `
      <h2>${post.title}</h2>
      <div class="time">${post.date}</div>
      ${post.html}
    `;
    entriesEl.appendChild(article);
  }
}

loadAll();
