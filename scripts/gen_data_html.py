"""把 数据对比总表.md 渲染成可直接在浏览器打开/打印的 HTML。

保持 md 为唯一数据源：改了 md 重跑本脚本即可。
支持：标题、表格、粗体、行内代码、引用块、分隔线、无序列表。
"""
import io
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / 'results' / '数据对比总表.md'
OUT = SRC.parent / '数据对比总表.html'


def inline(s):
    # 先处理行内代码，避免其中的 * 被当成粗体
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    return s


def convert(md):
    lines = md.split('\n')
    out, i = [], 0
    in_table = False
    in_list = False

    def close():
        nonlocal in_table, in_list
        if in_table:
            out.append('</table>')
            in_table = False
        if in_list:
            out.append('</ul>')
            in_list = False

    while i < len(lines):
        ln = lines[i]
        st = ln.strip()

        # 表格：当前行含 | 且下一行是分隔行
        if st.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:|-]+\|$', lines[i + 1].strip()):
            close()
            header = [c.strip() for c in st.strip('|').split('|')]
            align = [('r' if c.strip().endswith(':') and not c.strip().startswith(':')
                      else 'c' if c.strip().startswith(':') and c.strip().endswith(':')
                      else 'l') for c in lines[i + 1].strip().strip('|').split('|')]
            out.append('<table><thead><tr>')
            for h, a in zip(header, align):
                cls = f' class="a{a}"' if a != 'l' else ''
                out.append(f'<th{cls}>{inline(h)}</th>')
            out.append('</tr></thead><tbody>')
            in_table = True
            i += 2
            continue

        if in_table:
            if st.startswith('|'):
                cells = [c.strip() for c in st.strip('|').split('|')]
                out.append('<tr>')
                for j, c in enumerate(cells):
                    cls = ' class="a%s"' % {'r': 'r', 'c': 'c', 'l': 'l'}.get(
                        align[j] if j < len(align) else 'l', 'l')
                    out.append(f'<td{cls}>{inline(c)}</td>')
                out.append('</tr>')
                i += 1
                continue
            else:
                out.append('</tbody></table>')
                in_table = False

        if not st:
            close()
            i += 1
            continue

        if st.startswith('#### '):
            close(); out.append(f'<h4>{inline(st[5:])}</h4>')
        elif st.startswith('### '):
            close(); out.append(f'<h3>{inline(st[4:])}</h3>')
        elif st.startswith('## '):
            close(); out.append(f'<h2>{inline(st[3:])}</h2>')
        elif st.startswith('# '):
            close(); out.append(f'<h1>{inline(st[2:])}</h1>')
        elif st.startswith('---'):
            close(); out.append('<hr>')
        elif st.startswith('> '):
            close(); out.append(f'<blockquote>{inline(st[2:])}</blockquote>')
        elif re.match(r'^[-*] ', st):
            if not in_list:
                out.append('<ul>'); in_list = True
            out.append(f'<li>{inline(st[2:])}</li>')
        else:
            close()
            out.append(f'<p>{inline(st)}</p>')
        i += 1

    close()
    return '\n'.join(out)


def main():
    body = convert(io.open(SRC, encoding='utf8').read())
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>PCB 缺陷检测 · 数据对比总表</title>
<style>
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei", -apple-system, sans-serif; background:#f7f8fa; color:#1f2937;
       margin:0 auto; padding:36px 40px; max-width:1180px; line-height:1.7; }
h1 { color:#0f172a; font-size:28px; border-bottom:3px solid #3b82f6; padding-bottom:10px; margin-bottom:8px; }
h2 { color:#1e40af; font-size:20px; margin:36px 0 14px; padding-left:12px; border-left:4px solid #3b82f6; }
h3 { color:#1e3a8a; font-size:16px; margin:24px 0 10px; }
h4 { color:#334155; font-size:14.5px; margin:18px 0 8px; }
table { width:100%; border-collapse:collapse; margin:12px 0 22px; background:#fff;
        box-shadow:0 1px 3px rgba(0,0,0,.08); border-radius:8px; overflow:hidden; font-size:14px; }
th { background:#1e40af; color:#fff; padding:10px 12px; text-align:left; font-weight:600; font-size:13.5px; }
td { padding:9px 12px; border-top:1px solid #e5e7eb; }
tbody tr:nth-child(even) { background:#f9fafb; }
tbody tr:hover { background:#eff6ff; }
.a-r { text-align:right; font-family:Consolas,monospace; }
.a-c { text-align:center; }
strong { color:#b91c1c; }
code { background:#f1f5f9; color:#0f766e; padding:1px 6px; border-radius:4px;
       font-family:Consolas,monospace; font-size:13px; }
blockquote { margin:14px 0; padding:12px 18px; background:#fffbeb; border-left:4px solid #f59e0b;
             color:#78350f; font-size:13.5px; border-radius:0 6px 6px 0; }
hr { border:none; border-top:1px dashed #cbd5e1; margin:34px 0; }
ul { margin:10px 0; padding-left:24px; }
li { margin:4px 0; }
@media print {
  body { background:#fff; padding:0; }
  table { box-shadow:none; border:1px solid #cbd5e1; page-break-inside:avoid; }
  h2 { page-break-after:avoid; }
}
</style>
</head>
<body>
__BODY__
</body>
</html>
"""
    html = html.replace('__BODY__', body)
    io.open(OUT, 'w', encoding='utf8').write(html)
    print('written:', OUT)
    print('  tables =', html.count('<table>'), '| rows =', html.count('<tr>'))


if __name__ == '__main__':
    main()
