# Index
1) [JavaScript Basics](https://github.com/mertgulexe/vault/blob/main/others/README.md#1-javascript-basics)
2) [Ruby Basics](https://github.com/mertgulexe/vault/blob/main/others/README.md#2-ruby-basics)
3) [SQL Sandbox](https://github.com/mertgulexe/vault/blob/main/others/README.md#3-sql-sandbox)
4) [YAML Basics & Notes](https://github.com/mertgulexe/vault/blob/main/others/README.md#4-yaml-basics--notes)

# 1. JavaScript Basics
```js
// MDN (Mozilla Developer Network) is a good source for JS.
function changeMessages(message, discountRate) {
    document.getElementById('message').textContent = message;  // this is a DOM element
    document.getElementById("discount").textContent = discountRate;  // (Document Object Model)
}
// another fnc declaration method: let changeMessages = function() { }

let titleText = "Get a grip!",  // Unlike Python, variable names are written as camelCase.
    discount = 40,
    discountRate = "0.123abc"
    productName = "Hiking Boots",
    productRemovalYear = 2025;

discountRate = Number.parseFloat(discountRate)  // Variable updated.
var discountAvailable = !true  // 'var' is not used anymore because
// It has bugs. Its variable defined in a block still lives outside.
// Also, doesn't throw errors when defined wrongly etc.
let productId;  // undefined
productId = 612
productId = null  // value is wiped.
productId = undefined // undefined

const yearSold = 2022, // You have to set a constant to a value because 
      priceFrom_2021 = 39.99;  // declaration cannot be empty as 'let' above.
// initialPrice = 49.99  // Constant cannot be changed/updated anymore!

let discountText = `Up to ${discount}% off!`;  // Interpolation: Observe the backticks!
discountText = `Hey!


${discount}% off on ${productName.toUpperCase()}!`;

console.log(discountText, "with length:", discountText.length); 
/*
    html converts those new lines above to a whitespace but console log prints them.
*/

console.log(
    "'priceFrom_2021' variable type:", typeof priceFrom_2021,  // output: number
    "\nRemoval year:", ++productRemovalYear,  // productRemovalYear++ works as well but beware while using it.
    // ++yearSold  // Increment doesn't work for constants!
)

console.log(
    "The product was sold in " + yearSold.toString() + " before",
    `with the rate of ${discountRate}.`  // Note that "abc" is ignored while parsing above.
)

changeMessages(
    message = titleText,
    discount = discountText,
    );

let userProperties = {  // This is an object.
    firstName: "Mert",
    lastName: "Gül",
    age: 33.02,
    isMale: true
}

console.log(
    `User name: ${userProperties.firstName}\n`,
    typeof userProperties  // output: object (SIMILAR to "dictionaries" in Python, "hashes" in Ruby)
);


let price = "19.999"
price = Number.parseFloat(price);

if (price === NaN) {  // null, undefined, NaN are false.
    changeMessages("Not NaN Title", typeof price);
}

else if (+price.toFixed(2) === 20.00) {  // toFixed returns string. We added + as a prefix to make it an integer.
    changeMessages(
        `New Price: ${typeof +price.toFixed(1)}`,
        `Only ${price.toFixed(2)}`
        );  // btw, toFixed still thinks 'price' is a string :(
}

// ternary operator
let errorType = (typeof price === "number") ? "Number error": "Other errors";
console.log("Error type:", errorType);

/* Difference between '==' and '==='
    if (1 === "1"): false  ---> Different type! 'Strictly Equal to' or 'Identically Equal to'
    if (1 == "1"): true  ---> Converts the int to str.
*/

// loops
for (let i = 0; i <= 3; i++) {
    console.log("[INFO]:", i+1, "Mississippi")
}

let c = 5;
while (c > 0) {
    console.log("[INFO]:", c, "Mississippi");
    c--;
}

c2 = 10;
do {
    console.log("Count down from 10:", c2);  // Runs at least once.
    c2--;
} while (c2 >= 15);

// more of objects
let password = Symbol();
let person = {
    account_name: "Mert",
    age: 33,
    [password]: "123_!*psswrd",  // information hiding
    showInfo: function(some_var) {
        console.log(`You cannot retrieve ${this.account_name}'s password.`);
        console.log(some_var, "is printed on the console.");
    }
};
person.showInfo("Some Variable");
console.log(person.password);  // undefined
console.log(person[password]);  // 123_!*psswrd
```

# 2. Ruby Basics
```rb
=begin
These notes are taken from the Ruby tutorial in Codecademy.
The projects cover pretty much everything but probably not everything.
So, be careful when a recap is needed!
=end

### 1st Lesson: Introduction
# First Project: Putting the form in Formatter
print "What's your first name? "
first_name = gets.chomp
first_name.capitalize!

print "...and your last name? "
last_name = gets.chomp
last_name.capitalize!

print "Your city? "
city = gets.chomp
city.downcase!  # It's here for example.

print "State? "
state = gets.chomp
state.upcase!

print "#{first_name} #{last_name} is from #{city.capitalize!}, #{state}."


### 2nd Lesson: Control Flow
test_1 = (true || false) || (false && true)  # should be true
test_2 = (true && (3 < 4)) && (false || (0 == 0))  # should be true
test_3 = (false || false) || !(true || false)  # should be false

is_liar = false
print "Truth has been spoken!" unless is_liar
# Outputs: "Truth has been spoken!"

# Second Project: Duffy Duckifier
puts 3.times {puts "Welcome!"}

print "Provide an input: "

user_input = gets.chomp
user_input.downcase!

if user_input.include? "s"
  user_input.gsub!(/s/, "th")
  puts user_input
elsif user_input.length == 0
  puts "Please enter a string."
else
  puts "There is no 's' character(s) in the input string."
end


### 3nd Lesson: Looping
counter = 1
until counter > 10
  puts counter
  counter += 1
end

for num in 1...21
    puts num
  end

i = 20
loop {  # loop do ... end
  i -= 1
  next if i % 2 == 1
  print "#{i}"
  break if i <= 0
}

# Third Project: Redacted!
puts "Sentence: "
text = gets.chomp.downcase
puts "Redacted word: "
redact = gets.chomp.downcase
words = text.split(" ")

words.each do |w|
  if w == redact
    print "REDACTED "
  else
    print w + " "
  end
end


### 4th Lesson: Arrays & Hashes
friends = ["Milhouse", "Ralph", "Nelson", "Otto"]
family = {
    "Homer" => "dad",
    "Marge" => "mom",
    "Lisa" => "sister",
    "Maggie" => "sister"
}
friends.each { |x| puts "Friend #{x}" }
family.each { |x, y| puts "#{x}: #{y}" }

s = [  # 2D array
    ["ham", "swiss"],
    ["turkey", "cheddar"],
    ["roast beef", "gruyere"]
]
s.each do |sub_array|
  sub_array.each { |item|
    print item, " "
  }
end

sounds = Hash.new
sounds["dog"] = "woof"
sounds["cat"] = "meow"

secret_identities = {
  "The Batman" => "Bruce Wayne",
  "Superman" => "Clark Kent",
  "Wonder Woman" => "Diana Prince",
  "Freakazoid" => "Dexter Douglas"
}
secret_identities.each do |hero_alias, real_id|
  puts "#{hero_alias}: #{real_id}"
end

# Fourth Project: Create a Histogram
puts "Text: "
text = gets.chomp
words = text.split(" ")
frequencies = Hash.new(0)
# Py: freq = dict().fromkeys(words, 0)

words.each { |w|
  frequencies[w] += 1
}

frequencies = frequencies.sort_by { |w, c| c}
frequencies.reverse!
# Py: freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

frequencies.each do |w, c|
  puts w + " " + c.to_s  # or String(c)
end


### 5th Lesson: Blocks & Sorting
def array_of_10
  puts (1..10).to_a  # To array
end
array_of_10

def what_up(greeting, *friends)  # "splat arguments"
  friends.each { |friend| puts "#{greeting}, #{friend}!" }
end
what_up("What up", "Ian", "Zoe", "Zenas", "Eleanor")

book_1 = "A Wrinkle in Time"
book_2 = "A Brief History of Time"
print book_1 <=> book_2  # Output: 1. 
# This is called "Combined Comparison Operator". Outputs -1, 0, or 1.

books = [
  "Charlie and the Chocolate Factory",
  "War and Peace",
  "Utopia",
  "A Brief History of Time",
  "A Wrinkle in Time"
]
# To sort our books in ascending order, in-place
puts "Ascending order:"
books.sort! { |firstBook, secondBook| firstBook <=> secondBook }  # Loop block is unnecessary.
puts books
# Sort your books in descending order, in-place below
puts "\nDescending order:"
books.sort! do |firstBook, secondBook| secondBook <=> firstBook end
print books

# Fifth Project: Ordering Your Library
def sorter(arr, rev=false)
  unless rev
    return arr.sort!
  else
    return arr.sort.reverse!
  end
end

numbers = [3,7,2,5]
puts sorter(numbers, true)


### 6th Lesson: Hashes & Symbols
puts "string".object_id  # 16346560
puts "string".object_id  # 16346360
puts "string".class      # String
puts :symbol.object_id   # 802268
puts :symbol.object_id   # 802268
puts :symbol.class       # Symbol
# Why symbols?
# They’re immutable, meaning they can’t be changed once they’re created,
# Only one copy of any symbol exists at a given time, so they save memory,
# Symbol-as-keys are faster than strings-as-keys because of the above two reasons.

:some_symbol.to_s  # ==> "some_symbol"
"some_string".to_sym  # ==> :some_string
"some_string".intern  # ==> :some_string

strings = ["HTML", "CSS", "JavaScript", "Python", "Ruby"]
symbols = []
strings.each do |s|
  symbols.push(s.to_sym)
end
puts symbols  # [:HTML, :CSS, :JavaScript, :Python, :Ruby]

movies = {
  the_matrix: "Made in 1999"  # the key is still a symbol
}
puts movies  # {:the_matrix=>"Made in 1999"}

# Performance of String and Symbol
require 'benchmark'
string_AZ = Hash[("a".."z").to_a.zip((1..26).to_a)]
symbol_AZ = Hash[(:a..:z).to_a.zip((1..26).to_a)]
string_time = Benchmark.realtime do
  10_000_000.times { string_AZ["r"] }
end
symbol_time = Benchmark.realtime do
  10_000_000.times { symbol_AZ[:r] }
end
puts "String time: #{string_time} seconds."  # String time: 7.074 seconds.
puts "Symbol time: #{symbol_time} seconds."  # Symbol time: 4.989 seconds.

movie_ratings = {
  memento: 3,
  the_matrix: 5,
  truman_show: 4,
}
good_movies = movie_ratings.select { |k, v| v > 3}
puts good_movies  # {:the_matrix=>5, :truman_show=>4}

good_movies.each_key do |k| puts k end
good_movies.each_value do |v| puts v end

# Sixth Project: A Night At The Movies
movies = {fight_club: 5}
puts "What would you like to do?"
puts "-- Type 'add' to add a movie."
puts "-- Type 'update' to update a movie."
puts "-- Type 'display' to display all movies."
puts "-- Type 'delete' to delete a movie."

choice = gets.chomp
case choice
  when "add"
    puts "Movie name:"
    title = gets.chomp.to_sym
    if movies[title].nil?
      puts "Rating value:"
      rating = gets.chomp.to_i
      movies[title] = rating
      puts "The movie '#{title}' is added."
    else
      puts "The movie has already added to the database."
    end
  when "update"
    puts "Please provide the movie name to be updated:"
    title = gets.chomp.to_sym
    if movies[title].nil?
      puts "The movie is not in the database. Please add it first."
    else
      puts "New rating value:"
      updated_rating = gets.chomp.to_i
      movies[title] = updated_rating
      puts "The rating of the movie has been updated."
      puts movies
    end
  when "display"
    movies.each { |m, r| puts "#{m}: #{r}"}
  when "delete"
    puts "Which movie you would like to delete:"
    title = gets.chomp.to_sym
    if movies[title].nil?
      puts "No such movie exist in the database. Try again."
    else
      movies.delete(title)
      puts "The movie has been deleted from the database."
    end
  else
    puts "Error!"
end


### 7th Lesson: Refactoring
puts 3 < 4 ? "The condition is true." : "The condition is false."

puts "Hello there!"
greeting = gets.chomp
case greeting
  when "English" then puts "Hello!"
  when "French" then puts "Bonjour!"
  when "German" then puts "Guten Tag!"
  when "Finnish" then puts "Haloo!"
  else puts "I don't know that language!"
end
# Conditional Assignment:
favorite_book = nil
puts favorite_book
favorite_book ||= "Cat's Cradle"
puts favorite_book
favorite_book ||= "Why's (Poignant) Guide to Ruby"
puts favorite_book
favorite_book = "Why's (Poignant) Guide to Ruby"
puts favorite_book
# Outputs:
#  (nothing)
# Cat's Cradle
# Cat's Cradle
# Why's (Poignant) Guide to Ruby

# Implicit Return
def multiple_of_three(n)
  n % 3 == 0 ? "Yep!" : "Nope!"  # "return" is deleted!
end
puts multiple_of_three(5)  # Ruby’s methods will return the result of the last evaluated expression.

# Short-circuit evaluation
def a
  puts "A was evaluated!"
  return true
end
def b
  puts "B was also evaluated!"
  return true
end
puts a || b
puts "------"
puts a && b
# Outputs: (This also holds for 'false && ...')
# A was evaluated!
# true
# ------
# A was evaluated!
# B was also evaluated!
# true

"L".upto("P") {|l| print l + " "}  # L M N O P
print "\n"
10.downto(3) {|n| print n, " "}  # 10 9 8 7 6 5 4 3 

[1, 2, 3].respond_to?(:push)  # true
[1, 2, 3].respond_to?(:to_sym)  # false
age = 26
puts age.respond_to?(:next)  # true
puts age.next  # 27

# concatenation operator
alphabet = ["a", "b", "c"]
alphabet << "d"  # alphabet.push("d")

caption = "A giraffe surrounded by "
caption << "weezards!"  # caption += "weezards!"

# Seventh Project: The Refactor Factory
require 'prime'
def first_n_primes(n)  # "return"s and condition blocks are simplified
    "n must be an integer." unless n.is_a? Integer
    "n must be greater than 0." if n <= 0
    Prime.first n
end
puts first_n_primes(10)


### 8th Lesson: Blocks, Procs, And Lambdas
fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
doubled_fibs = fibs.collect {|i| i*2}  # .collect! for inplace
puts doubled_fibs  # [2, 2, 4, 6, 10, 16, 26, 42, 68, 110]
# yield:
def yield_name(name)
  puts "In the method! Let's yield."
  yield#("Jack")
  puts "In between the yields!"
  yield(name)
  puts "Block complete! Back in the method."
end
yield_name("Mert") do |n| puts "I am #{n}." end
# Outputs:
# In the method! Let's yield.
# I am .
# In between the yields!
# I am Mert.
# Block complete! Back in the method.

# Procs
floats = [1.2, 3.45, 0.91, 7.727, 11.42, 482.911]
round_down = Proc.new {|f| f.floor}
round_up = Proc.new do |f| f.ceil end
over_7 = Proc.new {|n| n >= 7}
ints = floats.collect(&round_down)
print ints, "\n"  # [1, 3, 0, 7, 11, 482]
print floats.map!(&round_up)    # [2, 4, 1, 8, 12, 483]
print floats.select(&over_7)  # [8, 12, 483]  # Note that the array is rounded up on the previous line.
# Note: The & is used to convert the cube proc into a block (since .collect!
# and .map! normally take a block). PS: collect & map are the same.
hi = Proc.new {puts "Hello!"}
hi.call
# You can also convert symbols to procs using '&'
numbers_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
strings_array = numbers_array.map(&:to_s)
# lambda
def lambda_demo(a_lambda)
  puts "I'm the method!"
  a_lambda.call
end
lambda_demo(lambda { puts "I'm the lambda!" })
# I'm the method!
# I'm the lambda!
# Lambda vs Proc: A lambda is just like a proc, only it cares about the number
# of arguments it gets and it returns to its calling method rather than returning immediately.
def batman_ironman_proc
  victor = Proc.new { return "Batman will win!" }
  victor.call
  "Iron Man will win!"
end
puts batman_ironman_proc
def batman_ironman_lambda
  victor = lambda { return "Batman will win!" }
  victor.call
  "Iron Man will win!"
end
puts batman_ironman_lambda
# Batman will win!
# Iron Man will win!


### 9th Lesson: OOP 1
class Computer
  $manufacturer = "Mango Computer, Inc."
  @@files = {hello: "Hello, world!"}
  def initialize(username, password)
    @username = username
    @password = password
  end
  def current_user
    @username
  end
  def self.display_files
    @@files
  end
end
# Make a new Computer instance:
hal = Computer.new("Dave", 12345)
puts "Current user: #{hal.current_user}"  # Current user: Dave
# @username belongs to the hal instance.
puts "Manufacturer: #{$manufacturer}"  # Manufacturer: Mango Computer, Inc.
# $manufacturer is global! We can get it directly.
puts "Files: #{Computer.display_files}"  # Files: {:hello=>"Hello, world!"}
# @@files belongs to the Computer class.

# Inheritance syntax
class BaseClass; end
class DerivedClass < BaseClass; end
# super
class Message
  @@messages_sent = 0
  def initialize(from, to)
    @from = from
    @to = to
    @@messages_sent += 1
  end
end
my_message = Message.new("f", "t")
class Email < Message
  def initialize(from, to)
    super
  end
end

# Nineth Project: Virtual Computer
class Machine
  @@users = {}
  def initialize(username, password)
    @username = username
    @password = password
    @@users[username] = password
    @files = {}
  end
  def create(filename)
    time = Time.now
    @files[filename] = time
    puts "#{filename} was created by #{@username} at #{time}."
  end
  def Machine.get_users
    @@users
  end
end
my_machine = Machine.new("eric", 01234)
your_machine = Machine.new("you", 56789)
my_machine.create("groceries.txt")
your_machine.create("todo.txt")
puts "Users: #{Machine.get_users}"
# groceries.txt was created by eric at 2022-02-23 13:46:19 +0000.
# todo.txt was created by you at 2022-02-23 13:46:19 +0000.
# Users: {"eric"=>668, "you"=>56789}


### 10th Lesson: OOP 2
class Asd
  public  # or private
  def name
    @name
  end
  def job=(new_job)  # convention. "hey, this method sets a value!"
    @job = new_job
  end
end
# attr_ for get/set
class Person
  attr_reader :name
  attr_accessor :job  # reader + writer = accessor
  # attr_writer :job
  def initialize(name, job)
    @name = name
    @job = job
  end
end
# Module: You can think of a module as a toolbox that contains a set methods and constants.
# You can think of modules as being very much like classes,
# only modules can’t create instances and can’t have subclasses.
# They’re just used to store things!
module Circle
  PI = 3.141592653589793  
  def Circle.area(radius)
    PI * radius**2
  end  
end
# '::' scope resolution operator
puts Math::PI

require "date"
puts Date.today
# "include" for Modules
class Angle
  include Math  # we want to use Math::cos but we don’t want to type Math::
  attr_accessor :radians
  def initialize(radians)
    @radians = radians
  end
  def cosine
    cos(@radians)
  end
end
acute = Angle.new(1)
puts acute.cosine  # 0.5403023058681398
# "extend": This means that class itself can use the methods, as opposed to instances of the class.
# ThePresent has a .now method that we'll extend to TheHereAnd
module ThePresent
  def now
    puts "It's #{Time.new.hour > 12 ? Time.new.hour - 12 + 3 :\
    Time.new.hour + 3}:#{Time.new.min} #{Time.new.hour > 12 ? 'PM' : 'AM'} (GMT+3)."
  end
end
class TheHereAnd
  extend ThePresent
end
TheHereAnd.now

# Tenth Project: Banking on Ruby
class Account
  attr_reader :name
  attr_reader :balance
  def initialize(name, balance=100)
    @name = name; @balance = balance; end
  private
  def pin
    @pin = 1234
  end
  private
  def pin_error
    "Access denied: incorrect PIN."
  end
  public
  def display_balance(pin_number)
    puts pin_number == pin ? "Balance: $#{balance}." : pin_error
  end
  public
  def withdraw(pin_number, amount)
    @balance -= amount
    if pin_number == pin
      puts "Withdrew #{amount}. New balance: $#{@balance}."
    else
      puts pin_error
    end
  end
end

my_account = Account.new("Mert", 12_770_000)
my_account.display_balance(1234)
my_account.withdraw(1234, 70000)
my_account.display_balance(1234)
my_account.withdraw(4321, 500)
```

# 3. SQL Sandbox
```sql
-- Example-1
SELECT
    customerNumber,
    customerName, 
    CONCAT(TRIM(BOTH ' ' FROM contactFirstName), ' ', contactLastName) AS contactName,
    UPPER(country), 
    ROUND(creditLimit, 0) AS creditLimit, 
    ROUND((creditLimit / 1.21), -2) AS creditLimit_Euro,
    creditLimit > 100000 AS isPremiumCustomer
    -- DISTINCT xyz    -- and there is this DISTINCT thing.
FROM
    classicmodels.customers
WHERE
    country NOT IN ('USA', 'Australia') AND creditLimit > 0
    -- WHERE country != 'USA'
ORDER BY customerNumber ASC
LIMIT 20;

-- Example-2
SELECT
    rating,
    -- these are called "aggregate functions":
    MAX(rental_duration),
    MIN(rental_rate),
    ROUND(AVG(length), 1) AS length_avg,
    SUM(replacement_cost)
FROM
    sakila.film
WHERE
    rental_rate > 1.99
GROUP BY 1    -- 1st column. Could write "rating". GROUP BY is placed between these two.
ORDER BY SUM(replacement_cost) DESC;

-- Example-3
SELECT
    salesRepEmployeeNumber,
    COUNT(*),    -- group by sütun ile gruplanan kayıtların satır sayısını verir.
    ROUND(AVG(creditLimit), 0) AS creditLimit_avg, 
    ROUND(MAX(creditLimit), 0) AS creditLimit_max,
    ROUND(MIN(creditLimit), 0) AS creditLimit_min
FROM
    classicmodels.customers
GROUP BY
    salesRepEmployeeNumber;

-- Example-4
SELECT
	productLine,
    productScale,
    COUNT(*) AS record_count
FROM
	classicmodels.products
GROUP BY
	1, 2
ORDER BY record_count DESC;
    
-- Example-5
SELECT 
    productLine,
    ROUND(AVG(buyPrice), 1) AS buyPrice_avg
FROM
    classicmodels.products
GROUP BY productLine
HAVING buyPrice_avg >= 47
    -- WHERE buyPrice_avg >= 48    -- 'where' cannot be used w/ aggregate funcs.
ORDER BY buyPrice_avg ASC;

-- Example-6
SELECT
    ord.orderNumber, 
    ord.orderDate, 
    cus.customerNumber,
    cus.contactLastName,
    cus.contactFirstName
FROM
    orders AS ord    -- "aliases' are used
JOIN
    customers cus ON ord.customerNumber = cus.customerNumber
LIMIT 20
;

-- Example-7
SELECT
    customerName, 
    contactLastName,
    contactFirstName, 
    country,
CASE country
    WHEN "USA"
        THEN "Hasan"
    WHEN "Canada"
        THEN "(Empty)"
    ELSE "Mert"
END "Representer"
FROM classicmodels.customers
;

-- Example-9
-- No such table as "STUDENT
-- Example output: "Student 3 has grade: A"
SELECT
    'Student ', ID, 'has grade: ',
    (CASE
        WHEN Score < 20 THEN 'F'
        WHEN Score >= 20 && Score < 40 THEN 'D'
        WHEN Score >= 40 && Score < 60 THEN 'C'
        WHEN Score >= 60 && Score < 80 THEN 'B'
        ELSE 'A'
    END) AS StudentGrade
FROM
    STUDENT;

-- Example-10:
-- No such tables.
-- Need some fix.
SELECT DISTINCT P.NAME, C.NAME
FROM SCHEDULE S
LEFT JOIN PROFESSOR P ON P.ID = S.PROFESSOR_ID
INNER JOIN COURSE C ON C.ID = S.COURSE_ID;

-- Example-11
SELECT c.phone, c.city, e.email, e.JobTitle
FROM classicmodels.customers c
RIGHT JOIN employees e ON c.salesRepEmployeeNumber=e.employeeNumber
WHERE c.phone OR c.city IS NOT NULL;

-- Example-12
CREATE TABLE my_t(
id INT PRIMARY KEY,
name VARCHAR(100) NOT NULL,
age INT DEFAULT NULL);
INSERT INTO my_t(id, name, age)
VALUES
		(0, 'mert', 32),
		(1, 'betül', 30),
        (2, 'erdem', 32);
ALTER TABLE my_t
ADD birth INT;
UPDATE my_t
SET birth = (2021 - age);  -- not working under safe update mode. Will be updated later.
```

# 4. YAML Basics & Notes
* These notes are from this [YT video](https://www.youtube.com/watch?v=1uFVr15xDGg). Also, [this website](https://www.tutorialspoint.com/yaml/index.htm) seems to provide some docs for basics.
* YAML is a data serialization language like XML, JSON.
* Same code in YAML, XML & JSON respectively:
	```yaml
    microservices:
        - app: user-authentication
		  port: 9000
		  version: 1.0
	```
	```xml
    <microservices>
		<microservice>
			<app>user-authentication</app>
			<port>9000</port>
			<version>1.0</version>
		</microservice>
	</microservices>
	```
	```json
	{
        "microservices": [
			{
				"app": "user-authentication",
				"port": 9000,
				"version": "1.0"
			}
		]
	}
	```
* YAML is more human readable & intuitive. It uses indentation and line separation. It is used in Docker Compose, Ansible, Prometheus, Kubernetes etc. 
* Basic syntax:
    ```yaml
    # simple key: value pairs
    app: user-authentication
    port: 9000
    version: 1.7
    ```
* Quotes are optional in the strings unless you are using special character like ``\n``.
* **Objects** to group things by indenting:
    * Note: You can use a YAML validator ([like this](https://onlineyamltools.com/edit-yaml)) since indentations could be hard to match.
    ```yaml
    microservice:  # this is the obj.
        app: user-authentication  # these are the attributes.
        port: 9000
        version: 1.7
        labels:  # object in an object.
            tag: nginx
    ```
* **Lists** & **lists in lists**.
    ```yaml
    microservice:
        - app: user-authentication  # 1st item in the list
          port: 9000
          version: 1.7  # Watch out for the indentation!
          deployed: true  # true/false, on/off, yes/no. All are valid.
        - app: shopping-cart  # 2nd item in the list
          ports:
            - 9002  # list in the list
            - 9003  # the validator gave me error when I removed the dash here.
            - 9004  # but not on this one. It could be a bug. This is the right way!
          versions: [1.9, "2.2", false, 3]  # for only primitive data types!
    
    other_microservice:
        - user-authentication
        - false  # could be such simple values as well
        
    ```
* Multi-line Strings
    * We use a pipe ``|`` character to do it. Like `\n` version in Python.
    * If we want yaml to interpret the string as a one-line sentence, we use ``>``.
        * It puts a whitespace by itself, don't leave one at the end.
    ```yaml
    multiLineString: |  # or > if one-liner
        this is a multiline string,
        and should be all on line
        really, okay?
    # output:
    # {
    #     "multiLineString": "this is a multiline string,\nand should be all on line\nreally, okay?\n"
    # }
    ```
    * `>-` or `|-` removes the linebreak (``\n``) appended at the end. It is called _block chomping indicator_. There are more crazy combinations here in this [stackoverflow](https://stackoverflow.com/questions/3790454/how-do-i-break-a-string-in-yaml-over-multiple-lines) post.
    * Real life example:
    ```yaml
    command:
        - sh
        - -c
        - |
            #!/usr/bin/env bash -e
            http () {
                local path="${1}"
                set -- -XGET -s --fail
                # some more stuff here
                curl -k "$@" "http://localhost:5601${path}"
            }
            http "/app/kibana"
    ```
* Environment variables (we do it with ``$`` sign.)
    ```yaml
    command:
        - /bin/sh
        - -ec
        - >-
            mysql -h 127.0.0.1 -u -root -p$MYSQL_ROOT_PASSWORD -e 'SELECT 1'
    ```
* Placeholders (its syntax is double curly braces ``{{ ... }}``)
    * This value get replaced using template generator.
    ```yml
    metadata:
        name: {{ .Values.service.name }}
    spec:
        selector:
            app: {{ .Values.service.app }}
        ports:
            - protocol: TCP
            port: {{ .Values.service.port }}
    ```
* If you wanna group your multiple yaml file into one, the syntax for it is three dashes `---`.
    ```yaml
    # simple key: value pairs
    app: user-authentication
    port: 9000
    version: 1.7
    ---
    microservice:  # this is the obj.
    app: user-authentication  # these are the attributes.
    port: 9000
    version: 1.7
    labels:  # object in an object.
        tag: nginx
    ```
