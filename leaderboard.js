document.addEventListener('DOMContentLoaded', () => {

    /**
     * Initializes all tab components on the page.
     * It finds all elements with '.tabs-container' and sets up click listeners
     * for each tab link within them to show the corresponding tab pane.
     */
    function initializeTabs() {
        const allTabsContainers = document.querySelectorAll('.tabs-container');

        allTabsContainers.forEach(container => {
            const tabLinks = container.querySelectorAll('.tab-link');
            const tabContent = container.querySelector('.tab-content');

            tabLinks.forEach(link => {
                link.addEventListener('click', () => {
                    const tabId = link.getAttribute('data-tab');
                    const targetPane = tabContent.querySelector(`#${tabId}`);

                    // Deactivate all links and panes within this specific container
                    tabLinks.forEach(item => item.classList.remove('active'));
                    tabContent.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

                    // Activate the clicked link and its corresponding pane
                    link.classList.add('active');
                    if (targetPane) {
                        targetPane.classList.add('active');
                    }
                });
            });
        });
    }

    /**
     * Initializes all sortable tables on the page.
     * It handles sorting by column click and applies a color gradient to cells.
     */
    function initializeSortableTables() {
        /**
         * Applies a blue gradient background to cells in a column based on their numeric value.
         * @param {HTMLTableElement} table - The table element to process.
         * @param {number} columnIndex - The index of the column to color.
         */
        function applyColumnColoring(table, columnIndex) {
            const tbody = table.querySelector('tbody');
            if (!tbody) return;

            const cells = Array.from(tbody.querySelectorAll(`tr:not(.group-header) > td:nth-child(${columnIndex + 1})`));
            const values = cells.map(cell => parseFloat(cell.textContent)).filter(v => !isNaN(v));

            if (values.length < 2) return;

            const maxVal = Math.max(...values);
            const minVal = Math.min(...values);
            const range = maxVal - minVal;

            cells.forEach(cell => {
                const value = parseFloat(cell.textContent);
                if (isNaN(value)) return;

                const normalized = (range === 0) ? 1 : (value - minVal) / range;
                const lightness = 97 - (normalized * 37); // Interpolate lightness from 97% to 60%
                cell.style.backgroundColor = `hsl(208, 100%, ${lightness}%)`;
            });
        }

        /**
         * Sorts a table by a specific column, preserving row groups.
         * @param {HTMLTableElement} table - The table element to sort.
         * @param {number} columnIndex - The index of the column to sort by.
         */
        function sortTableByColumn(table, columnIndex) {
            const tbody = table.querySelector('tbody');
            if (!tbody) return;

            const groups = new Map();
            let currentHeader = null;

            Array.from(tbody.rows).forEach(row => {
                if (row.classList.contains('group-header')) {
                    currentHeader = row;
                    groups.set(currentHeader, []);
                } else if (currentHeader) {
                    groups.get(currentHeader).push(row);
                }
            });

            for (const rows of groups.values()) {
                rows.sort((a, b) => {
                    const aText = a.cells[columnIndex].textContent.trim();
                    const bText = b.cells[columnIndex].textContent.trim();
                    if (aText === '-' || bText === '-') return aText === '-' ? 1 : -1;
                    
                    const aNum = parseFloat(aText);
                    const bNum = parseFloat(bText);
                    return !isNaN(aNum) && !isNaN(bNum) ? bNum - aNum : bText.localeCompare(aText);
                });
            }

            tbody.innerHTML = '';
            for (const [header, rows] of groups.entries()) {
                tbody.appendChild(header);
                rows.forEach(row => tbody.appendChild(row));
            }
        }

        /**
         * Calculates the true visual column index for a header cell, accounting for colspan/rowspan.
         * @param {HTMLTableCellElement} headerCell - The <th> element.
         * @returns {number} The calculated visual column index.
         */
        function getTrueColumnIndex(headerCell) {
            const table = headerCell.closest('table');
            const headerRows = Array.from(table.tHead.rows);
            const matrix = Array.from({ length: headerRows.length }, () => Array(50).fill(null)); // Assume max 50 columns

            headerRows.forEach((row, rowIndex) => {
                let matrixColIndex = 0;
                Array.from(row.cells).forEach(cell => {
                    while (matrix[rowIndex][matrixColIndex]) {
                        matrixColIndex++;
                    }
                    const colSpan = parseInt(cell.getAttribute('colspan') || '1');
                    const rowSpan = parseInt(cell.getAttribute('rowspan') || '1');
                    for (let r = 0; r < rowSpan; r++) {
                        for (let c = 0; c < colSpan; c++) {
                            if (matrix[rowIndex + r]) {
                                matrix[rowIndex + r][matrixColIndex + c] = cell;
                            }
                        }
                    }
                    matrixColIndex += colSpan;
                });
            });

            const rowIndex = headerCell.parentElement.rowIndex;
            return matrix[rowIndex] ? matrix[rowIndex].indexOf(headerCell) : -1;
        }

        document.querySelectorAll('.sortable-table').forEach(table => {
            const sortableHeaders = table.querySelectorAll('thead th:not([colspan])');
            
            // Initial coloring for all sortable columns
            sortableHeaders.forEach(headerCell => {
                const trueColumnIndex = getTrueColumnIndex(headerCell);
                if (trueColumnIndex !== -1) {
                    applyColumnColoring(table, trueColumnIndex);
                }
            });
            
            // Add click listeners for sorting
            sortableHeaders.forEach(headerCell => {
                headerCell.addEventListener('click', () => {
                    const trueColumnIndex = getTrueColumnIndex(headerCell);
                    if (trueColumnIndex !== -1) {
                        sortTableByColumn(table, trueColumnIndex);
                    }
                });
            });
        });
    }

    // --- INITIALIZE ALL COMPONENTS ---
    initializeTabs();
    initializeSortableTables();
});