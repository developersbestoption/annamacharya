// ================= LOGIN =================

const loginForm = document.getElementById("loginForm");

loginForm.addEventListener("submit", function(event) {

    event.preventDefault();

    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;

    // Temporary frontend login
    // Later PHP + MySQL will handle this.

    if (email === "admin@gmail.com" && password === "123456") {

        document.getElementById("loginPage")
            .classList.add("hidden");

        document.getElementById("app")
            .classList.remove("hidden");

    } else {

        alert("Invalid email or password!");

    }

});


// ================= LOGOUT =================

function logout() {

    document.getElementById("app")
        .classList.add("hidden");

    document.getElementById("loginPage")
        .classList.remove("hidden");

    document.getElementById("loginForm").reset();

}


// ================= PAGE NAVIGATION =================

function showPage(pageName, button = null) {

    // Hide all pages

    const pages = document.querySelectorAll(".page");

    pages.forEach(function(page) {
        page.classList.add("hidden");
    });


    // Show selected page

    document.getElementById(pageName)
        .classList.remove("hidden");


    // Change page title

    const titles = {

        dashboard: "Dashboard",
        books: "Books",
        issues: "Issues & Returns",
        payments: "Payments",
        users: "Users"

    };

    document.getElementById("pageTitle")
        .textContent = titles[pageName];


    // Remove active class

    const navItems = document.querySelectorAll(".nav-item");

    navItems.forEach(function(item) {
        item.classList.remove("active");
    });


    // Add active class

    if (button) {
        button.classList.add("active");
    }

}


// ================= SEARCH BOOKS =================

function searchBooks() {

    const input =
        document.getElementById("bookSearch");

    const searchText =
        input.value.toLowerCase();

    const rows =
        document.querySelectorAll("#bookTable tbody tr");


    rows.forEach(function(row) {

        const bookName =
            row.cells[1].textContent.toLowerCase();

        const author =
            row.cells[2].textContent.toLowerCase();

        const category =
            row.cells[3].textContent.toLowerCase();


        if (
            bookName.includes(searchText) ||
            author.includes(searchText) ||
            category.includes(searchText)
        ) {

            row.style.display = "";

        } else {

            row.style.display = "none";

        }

    });

}


// ================= ADD BOOK MODAL =================

function openAddBook() {

    document.getElementById("bookModal")
        .classList.remove("hidden");

}


function closeAddBook() {

    document.getElementById("bookModal")
        .classList.add("hidden");

}


// ================= ADD BOOK =================

function addBook(event) {

    event.preventDefault();

    const name =
        document.getElementById("newBookName").value;

    const author =
        document.getElementById("newAuthor").value;

    const category =
        document.getElementById("newCategory").value;

    const copies =
        document.getElementById("newCopies").value;


    const table =
        document.querySelector("#bookTable tbody");


    const newRow =
        document.createElement("tr");


    newRow.innerHTML = `

        <td>NEW</td>

        <td>${name}</td>

        <td>${author}</td>

        <td>${category}</td>

        <td>${copies}</td>

        <td>${copies}</td>

        <td>
            <span class="badge available">
                Available
            </span>
        </td>

    `;


    table.appendChild(newRow);


    closeAddBook();


    document.querySelector("#bookModal form").reset();


    alert("Book added successfully!");

}