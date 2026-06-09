# ## 10. Affichage de la réponse
# 
# La fonction `display_response()` produit la sortie structurée visible dans les images de référence.

# In[ ]:


from IPython.display import display, HTML

def display_response(result: dict):
    """Affiche la réponse avec le rendu visuel structuré (cartes + badges articles)."""

    question       = result["question"]
    raw            = result["reponse"]
    articles_cites = result["articles_cites"]

    # Nettoie le prompt répété (sécurité supplémentaire)
    for marker in ["Réponse :", "## 1."]:
        if marker in raw:
            idx = raw.find(marker)
            raw = raw[idx:].strip()
            break


    # Parse les sections ## X. Titre  (NOUVEAU)
    import re as _re
    sections = _re.split(r"(## \d+\..*)", raw)
    parsed   = []
    i = 0
    while i < len(sections):
        s = sections[i].strip()
        if _re.match(r"## \d+\.", s) and i + 1 < len(sections):
            title   = _re.sub(r"## \d+\.\s*", "", s).strip()
            content = sections[i + 1].strip()
            parsed.append((title, content))
            i += 2
        else:
            i += 1

    # ── NOUVEAU : si le LLM n'a pas respecté la structure ──────────────
    if not parsed:
        # Affiche la réponse brute dans une seule carte neutre
        parsed = [("Réponse", raw)]

    # Couleurs par section
    dot_colors = ["#639922", "#185FA5", "#7F77DD", "#BA7517", "#0F6E56"]

    # ── Construction HTML ──
    cards_html = ""

    # Métriques (uniquement pour les réponses RAG sémantique, pas les lookups directs)
    if len(articles_cites) > 0 and "Trouvé directement" not in raw:
        cards_html += f"""
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px">
          <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px 12px">
            <p style="font-size:11px;color:var(--color-text-tertiary);margin:0 0 2px">Variantes générées</p>
            <p style="font-size:22px;font-weight:500;margin:0;color:var(--color-text-primary)">4</p>
          </div>
          <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px 12px">
            <p style="font-size:11px;color:var(--color-text-tertiary);margin:0 0 2px">Articles récupérés</p>
            <p style="font-size:22px;font-weight:500;margin:0;color:var(--color-text-primary)">8</p>
          </div>
          <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px 12px">
            <p style="font-size:11px;color:var(--color-text-tertiary);margin:0 0 2px">Après re-ranking</p>
            <p style="font-size:22px;font-weight:500;margin:0;color:var(--color-text-primary)">{len(articles_cites)}</p>
          </div>
        </div>
        """

    # Sections de réponse
    for idx, (title, content) in enumerate(parsed):
        dot   = dot_colors[idx] if idx < len(dot_colors) else "#888"
        # Mise en forme spéciale pour vigilance (amber) et recommandation (green)
        if idx == 3:  # Points de vigilance
            inner = f'<div style="background:#faeeda;border:0.5px solid #BA7517;border-radius:8px;padding:10px 14px;font-size:14px;color:#633806;line-height:1.65">{content}</div>'
        elif idx == 4:  # Recommandation
            inner = f'<div style="background:#eaf3de;border:0.5px solid #3B6D11;border-radius:8px;padding:10px 14px;font-size:14px;color:#27500A;line-height:1.65">{content}</div>'
        else:
            inner = f'<p style="font-size:14px;color:var(--color-text-primary);line-height:1.65;margin:0">{content}</p>'

        cards_html += f"""
        <div style="background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:12px;padding:1rem 1.25rem;margin-bottom:10px">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <div style="width:8px;height:8px;border-radius:50%;background:{dot};flex-shrink:0"></div>
            <span style="font-size:13px;font-weight:500;color:var(--color-text-primary)">{idx+1} — {title}</span>
          </div>
          {inner}
        </div>
        """

    # Badges articles cités
    if articles_cites:
        badges = ""
        for art in articles_cites:
            # Extrait "Article X" et le titre du chapitre
            m = _re.match(r"(Article \S+)\s*\((.+?)\)", art)
            if m:
                num   = m.group(1)
                chap  = m.group(2)
                label = num
            else:
                label = art
                chap  = ""
            badges += f'<span style="display:inline-flex;align-items:center;gap:6px;background:var(--color-background-secondary);border:0.5px solid var(--color-border-tertiary);border-radius:20px;padding:4px 12px;font-size:12px;color:var(--color-text-secondary);margin:3px"><span style="font-weight:500;color:var(--color-text-info)">{label}</span>{chap}</span>'

        cards_html += f"""
        <div style="background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:12px;padding:1rem 1.25rem;margin-bottom:10px">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <div style="width:8px;height:8px;border-radius:50%;background:#185FA5;flex-shrink:0"></div>
            <span style="font-size:13px;font-weight:500;color:var(--color-text-primary)">Articles RGPD mobilisés</span>
          </div>
          {badges}
        </div>
        """

    # Avertissement
    cards_html += '<p style="font-size:12px;color:var(--color-text-tertiary);margin:8px 0 0">⚠️ Cette réponse est informative et ne constitue pas un avis juridique.</p>'

    html_out = f"""
    <div style="font-family:sans-serif;padding:4px 0">
      <p style="font-size:13px;color:var(--color-text-secondary);margin:0 0 6px">Question posée</p>
      <div style="display:inline-block;background:var(--color-background-secondary);border:0.5px solid var(--color-border-tertiary);border-radius:20px;padding:6px 16px;font-size:13px;color:var(--color-text-secondary);margin-bottom:14px">{question}</div>
      {cards_html}
    </div>
    """
    display(HTML(html_out))


print("✅  display_response() prête")
