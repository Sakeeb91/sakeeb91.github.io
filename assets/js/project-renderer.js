(function () {
    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text) node.textContent = text;
        return node;
    }

    function createLink(link) {
        const anchor = el('a');
        anchor.href = link.url;
        anchor.target = '_blank';
        anchor.rel = 'noopener';
        anchor.textContent = link.label.toUpperCase();
        return anchor;
    }

    function createProjectCard(project) {
        const card = el('article', 'project');

        if (Array.isArray(project.tags) && project.tags.length) {
            const meta = el('div', 'project-meta');
            project.tags.forEach(tag => {
                meta.appendChild(el('span', '', tag));
            });
            card.appendChild(meta);
        }

        card.appendChild(el('h3', 'project-title', project.title));
        card.appendChild(el('p', 'project-subtitle', project.subtitle));

        if (Array.isArray(project.highlights) && project.highlights.length) {
            const ul = el('ul', 'project-highlights');
            project.highlights.forEach(item => {
                const li = document.createElement('li');
                const label = el('span', '');
                label.textContent = item.label;
                li.appendChild(label);
                li.appendChild(document.createTextNode(item.text));
                ul.appendChild(li);
            });
            card.appendChild(ul);
        }

        if (Array.isArray(project.links) && project.links.length) {
            const links = el('div', 'project-links');
            project.links.forEach(link => links.appendChild(createLink(link)));
            card.appendChild(links);
        }

        return card;
    }

    window.renderProjectGrid = function renderProjectGrid(selector, { limit } = {}) {
        const container = typeof selector === 'string' ? document.querySelector(selector) : selector;
        if (!container || !Array.isArray(window.PROJECTS)) return;

        const items = typeof limit === 'number' ? window.PROJECTS.slice(0, limit) : window.PROJECTS;
        container.innerHTML = '';
        items.forEach(project => container.appendChild(createProjectCard(project)));
    };
})();
