// Validate LaTeX snippets with KaTeX. stdin: JSON [{tex, display}], stdout: JSON [{i, msg}]
const katex = require(process.argv[2]);
let data = "";
process.stdin.on("data", (d) => (data += d)).on("end", () => {
  const errs = [];
  JSON.parse(data).forEach((it, i) => {
    try {
      katex.renderToString(it.tex, { displayMode: it.display, throwOnError: true, strict: false });
    } catch (e) {
      errs.push({ i, msg: e.message });
    }
  });
  console.log(JSON.stringify(errs));
});
