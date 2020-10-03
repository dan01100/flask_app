var names = ["Alex Oliveira", "Danielle Taylor", "Matthew Riddle", "Jeff Curran", "Glover Teixeira", "Fabricio Werdum", "Paddy Holohan", "James Irvin", "Paul Daley", "Ryan Couture", "Rameau Thierry Sokoudjou", "Bethe Correia", "Heath Herring", "Ivan Salaverry", "Matt Hamill", "Austin Hubbard", "Nathaniel Wood", "Remco Pardoel", "Hannah Cifers", "John Howard", "Montana De La Rosa", "Aleksei Oleinik", "Max Griffin", "Vitor Miranda", "Mark Hominick", "Kalib Starnes", "Darren Elkins", "Roland Delorme", "Joey Beltran", "Guido Cannetti", "Luis Henrique", "John Dodson", "Eddie Wineland", "Shawn Jordan", "Anthony Smith", "Jess Liaudin", "Volkan Oezdemir", "Nate Quarry", "Jeff Monson", "Dokonjonosuke Mishima", "Justin Gaethje", "Cyrille Diabate", "Hatsu Hioki", "Tai Tuivasa", "Cain Velasquez", "Johnny Case", "Alexander Otsuka", "Dustin Kimura", "Sam Sicilia", "George Sotiropoulos", "Benson Henderson", "Carla Esparza", "Michael Johnson", "Luke Cummo", "Jon Jones", "Belal Muhammad", "Abdul Razak Alhassan", "Aljamain Sterling", "Dominick Reyes", "Vagner Rocha", "Josh Thomson", "Michelle Waterson", "Nasrat Haqparast", "Henry Cejudo", "Yoshihiro Akiyama", "Ryan Jimmo", "Stipe Miocic", "Rob Kimmons", "Mario Sperry", "Ray Borg", "Bobby Southworth", "Kai Kara-France", "Pete Williams", "Phil Harris", "Alexander Volkov", "Diego Brandao", "Daniel Pineda", "Kevin Randleman", "David Heath", "Soa Palelei", "Gerald Harris", "Mike Russow", "Damian Stasiak", "Jimmie Rivera", "Kenan Song", "Jason Von Flue", "Hacran Dias", "Matt Grice", "Ed Herman", "Jared Gordon", "Roan Carneiro", "Caol Uno", "Shonie Carter", "Niko Price", "Mark Coleman", "Marcin Tybura", "Vicente Luque", "Justin Buchholz", "Jessica Penne", "Michinori Tanaka", "Roger Huerta", "Spencer Fisher", "Nina Ansaroff", "Joey Villasenor", "Katsunori Kikuno", "Kazuyuki Fujita", "Shannon Gugerty", "Francimar Barroso", "Drew Dober", "Jose Aldo", "Weili Zhang", "Yoshiyuki Yoshida", "Steven Peterson", "Daniel Teymur", "Alex Stiebling", "Anthony Njokuani", "Darren Uyenoyama", "Gunnar Nelson", "Tra Telligman", "Robert Whiteford", "Bob Sapp", "Richard Walsh", "Nikita Krylov", "Norman Parke", "Luiz Azeredo", "Richard Crunkilton", "Hiroyuki Takaya", "Sergio Pettis", "Renato Moicano", "Donald Cerrone", "Daniel Omielanczuk", "Chris Beal", "Ramazan Emeev", "Henrique da Silva", "Kevin Ferguson", "Charles Bennett", "Antonio Carlos Junior", "Rashid Magomedov", "Alejandro Perez", "Fabio Maldonado", "Cung Le", "Brett Rogers", "Tom Breese", "Renan Barao", "Alex Morono", "TJ Dillashaw", "Chris Lytle", "Neil Magny", "Ricky Simon", "Rafael Natal", "Rich Attonito", "Angela Hill", "Brian Bowles", "Kimo Leopoldo", "Bobby Voelker", "Mark Bocek", "Alex Karalexis", "Markus Perez", "Magnus Cedenblad", "Jordan Rinaldi", "Jesse Taylor", "Islam Makhachev", "Charlie Valencia", "Alex Chambers", "Robert Whittaker", "Jon Madsen", "Seth Baczynski", "Nicholas Musoke", "Brian Stann", "Pedro Munhoz", "Renato Sobral", "Shane Burgos", "Gina Mazany", "Dylan Andrews", "Luke Stewart", "Ricardo Arona", "Hugo Viana", "Bec Rawlings", "Devin Cole", "David Loiseau", "Valentijn Overeem", "John Phillips", "Nick Diaz", "Frank Mir", "Artem Lobov", "Joachim Hansen", "Dean Lister", "Scott Holtzman", "Cheick Kongo", "Terry Etim", "Bart Palaszewski", "Juliana Lima", "Alexander Volkanovski", "Neil Seery", "Scott Ferrozzo", "Michael McDonald", "Ovince Saint Preux", "Danny Martinez", "Geoff Neal", "Tony Fryklund", "Chas Skelly", "Dong Hyun Kim", "Shane Campbell", "Aaron Simpson", "Jacob Volkmann", "Deiveson Figueiredo", "Johnny Bedford", "Scott Jorgensen", "Mu Bae Choi", "Alessio Di Chirico", "Rick Story", "Anderson Silva", "Drew Fickett", "Chad Mendes", "Kyoji Horiguchi", "Matt Hughes", "Jorge Gurgel", "Macy Chiasson", "Jack Hermansson", "Jessica Andrade", "Dan Severn", "Travis Lutter", "Peter Sobotta", "Takeya Mizugaki", "Marvin Eastman", "Frankie Edgar", "Pat Healy", "Stefan Struve", "Junior Albini", "Miesha Tate", "Ilir Latifi", "Evan Tanner", "Zach Makovsky", "Felice Herrig", "Rashad Evans", "Tim Kennedy", "John Hathaway", "Dustin Pague", "Jason Black", "Bobby Stack", "Frankie Saenz", "Brian Ebersole", "Jay Hieron", "Marion Reneau", "Jeremy Stephens", "Mike Kyle", "Gadzhimurad Antigulov", "Justin Ledet", "Magomed Mustafaev", "Nicolas Dalby", "Masaaki Satake", "Brendan Schaub", "Kyle Bochniak", "Royce Gracie", "Iuri Alcantara", "Timothy Johnson", "Shane Roller", "Anthony Rocco Martin", "Elizeu Zaleski dos Santos", "Evangelista Santos", "Dustin Hazelett", "Marco Polo Reyes", "Teruto Ishihara", "Chico Camus", "Hermes Franca", "Shana Dobson", "Paulo Filho", "Demian Maia", "Assuerio Silva", "Tatsuya Mizuno", "Yuta Sasaki", "Yaotzin Meza", "Valentina Shevchenko", "Carlo Prater", "Din Thomas", "Ian Freeman", "Zak Cummings", "Chris Gruetzemacher", "Jon Fitch", "Brian Foster", "Gan McGee", "Luiz Cane", "Nick Ring", "Cody Garbrandt", "Pat Miletich", "Brad Scott", "Jake Matthews", "Raquel Pennington", "Hayato Sakurai", "Lumumba Sayers", "Alex Garcia", "Jason Brilz", "Ed Ratcliff", "Alessio Sakara", "Andrew Craig", "Dave Herman", "Ricardo Lamas", "Alistair Overeem", "Joe Proctor", "Steve Cantwell", "Mark De La Rosa", "Mairbek Taisumov", "Alvin Cacdac", "Megan Anderson", "Kevin Aguilar", "Justin Eilers", "Thiago Tavares", "Nick Hein", "Charles Oliveira", "Brandon Thatch", "Henry Briones", "Jairzinho Rozenstruik", "Matt Lindland", "Mike Massenzio", "Diego Sanchez", "Sean Pierson", "Guy Mezger", "Sage Northcutt", "Tim Means", "Keita Nakamura", "Gabriel Benitez", "Chase Sherman", "Shane Carwin", "Daijiro Matsui", "Antonio Banuelos", "Mike Pyle", "Colby Covington", "Tatsuya Kawajiri", "Paulo Thiago", "Sam Alvey", "Riki Fukuda", "Sean Sherk", "Paige VanZant", "Marvin Vettori", "Roosevelt Roberts", "Chris Wade", "Cynthia Calvillo", "Andre Winner", "Jason Lambert", "Kendall Grove", "Francis Carmont", "Kuniyoshi Hironaka", "Dennis Bermudez", "Ali Bagautinov", "TJ Waldburger", "Jose Quinonez", "Dave Beneteau", "Mauricio Rua", "Germaine de Randamie", "Dennis Hallman", "Stephen Thompson", "Manny Tapia", "Don Frye", "Christos Giagos", "James Te Huna", "Gabriel Gonzaga", "Quinn Mulhern", "Mickey Gall", "Joe Lauzon", "Ian McCall", "Lyle Beerbohm", "Tiequan Zhang", "Joe Doerksen", "Bibiano Fernandes", "Israel Adesanya", "Hakeem Dawodu", "Cole Miller", "Lavar Johnson", "Mac Danzig", "Mark Munoz", "Shayna Baszler", "Dan Henderson", "Alex Serdyukov", "Dongi Yang", "Aaron Rosa", "Gilbert Yvel", "Kenji Osawa", "Anthony Johnson", "Marloes Coenen", "Efrain Escudero", "Demetrious Johnson", "Cristiane Justino", "Estevan Payan", "Karl Roberson", "Charlie Brenneman", "Cristiano Marcello", "Vinc Pichel", "David Branch", "Ben Saunders", "Brad Pickett", "Julie Kedzie", "Joanna Jedrzejczyk", "Tonya Evinger", "Nah-Shon Burrell", "Ryan LaFlare", "Naoya Ogawa", "Erik Perez", "Justin Scoggins", "Gary Goodridge", "Jamie Varner", "Kevin Casey", "Yoshiro Maeda", "Thomas Almeida", "Francisco Trinaldo", "Mara Romero Borella", "Desmond Green", "Ben Nguyen", "Shinya Aoki", "Ron Waterman", "Derek Brunson", "Jorge Santiago", "Valerie Letourneau", "Tony DeSouza", "Alan Belcher", "Eric Lawson", "Cole Escovedo", "Alexandre Pantoja", "Brock Larson", "Jimy Hettes", "Nathan Coy", "Ryan Benoit", "Holly Holm", "Louis Smolka", "Xiaonan Yan", "Jens Pulver", "Maurice Greene", "Vitor Belfort", "Sarah Kaufman", "Kyle Kingsbury", "Patrick Smith", "Jordan Mein", "Clint Hester", "Roger Gracie", "Francisco Rivera", "Jake Ellenberger", "Ryo Chonan", "Josh Samman", "Gray Maynard", "Jake Collier", "Jason Knight", "Douglas Silva de Andrade", "LC Davis", "Bryan Caraway", "Curtis Millender", "Mitch Gagnon", "Mackenzie Dern", "Andre Galvao", "Ashley Yoder", "Kurt Pellegrino", "Thiago Santos", "Dong Sik Yoon", "Greg Hardy", "Sergio Moraes", "Kyle Noke", "Tiki Ghosn", "Ebenezer Fontes Braga", "Tamdan McCrory", "Ryan Jensen", "Ildemar Alcantara", "Tyron Woodley", "Rousimar Palhares", "Phillipe Nover", "Eddie Alvarez", "Mike Pierce", "Alan Jouban", "Murilo Bustamante", "Alex Perez", "Fabiano Iha", "Hiromitsu Miura", "Diego Nunes", "Brett Johns", "Petr Yan", "Warlley Alves", "Kazushi Sakuraba", "Melvin Manhoef", "Carlos Newton", "Alexander Yakovlev", "Patrick Cummins", "Rich Franklin", "Alex Caceres", "Vladimir Matyushenko", "Magomed Ankalaev", "Dan Bobish", "Nordine Taleb", "Cyril Asker", "Krzysztof Soszynski", "Pedro Rizzo", "Akihiro Gono", "Jacare Souza", "Sean Strickland", "Chris Leben", "Phil Davis", "Roxanne Modafferi", "Andy Ogle", "Brad Tavares", "Randy Couture", "Allan Goes", "Ross Pearson", "Leandro Silva", "Muhammed Lawal", "James Thompson", "Bartosz Fabinski", "Rony Jason", "Jack Marshman", "Thiago Silva", "Piotr Hallmann", "Joseph Benavidez", "Rani Yahya", "Paul Buentello", "Jimi Manuwa", "Alex White", "Bojan Velickovic", "Daniel Sarafian", "James Vick", "Eiji Mitsuoka", "Paulo Cesar Silva", "Andrew Holbrook", "Jonathan Martinez", "Albert Morales", "Brad Blackburn", "Hirotaka Yokoi", "Denis Kang", "Ronny Markes", "Veronica Macedo", "Michael Bisping", "Tsuyoshi Kohsaka", "Kazuhiro Nakamura", "Tim Sylvia", "Edgar Garcia", "Shungo Oyama", "Myles Jury", "Martin Kampmann", "Wanderlei Silva", "KJ Noons", "Amanda Cooper", "Jake Shields", "Krzysztof Jotko", "Tom Lawlor", "Arnold Allen", "Danny Roberts", "Maurice Smith", "Jared Cannonier", "Siyar Bahadurzada", "Dave Menne", "Charles Rosa", "Jennifer Maia", "Rustam Khabilov", "Al Iaquinta", "Daiju Takase", "Gregor Gillespie", "Kevin Lee", "Todd Duffee", "Katlyn Chookagian", "Keith Jardine", "Steven Siler", "Dennis Siver", "Enson Inoue", "Chris Weidman", "Andre Soukhamthath", "Daron Cruickshank", "Ronda Rousey", "Andrew Sanchez", "Rose Namajunas", "Sam Stout", "Elias Silverio", "Marc Diakiese", "Justin Edwards", "Jared Hamman", "Garreth McLellan", "Khabib Nurmagomedov", "Elvis Sinosic", "Michel Prazeres", "Norifumi Yamamoto", "Abel Trujillo", "Matt Mitrione", "Wagnney Fabiano", "Damir Hadzovic", "Dooho Choi", "Daisuke Nakamura", "Alexis Davis", "Josh Barnett", "Joe Stevenson", "Rodrigo Damm", "Sean O'Malley", "Andre Fili", "Manny Bermudez", "Isaac Vallie-Flagg", "Gilbert Burns", "Albert Tumenov", "Antonio Rodrigo Nogueira", "Pat Barry", "Christian Wellisch", "Ricco Rodriguez", "Luis Pena", "Dustin Ortiz", "Muslim Salikhov", "Luke Sanders", "Matthew Lopez", "Alan Patrick", "Luiz Firmino", "Mirsad Bektic", "Makoto Takimoto", "Yoshihisa Yamamoto", "Chad Griggs", "Matt Brown", "Ramsey Nijem", "Marina Rodriguez", "Nik Lentz", "John Albert", "Kyung Ho Kang", "Dan Hooker", "Marcus Brimage", "Renzo Gracie", "Marco Ruas", "Tyson Griffin", "Mark Kerr", "Chad Laprise", "Frank Shamrock", "Emily Whitmire", "JJ Aldrich", "Karlos Vemola", "Liz Carmouche", "Daniel Roberts", "Joanne Calderwood", "Tito Ortiz", "Zelg Galesic", "Will Brooks", "Trevor Prangley", "Marcus Aurelio", "Thibault Gouti", "James Krause", "Yushin Okami", "Jason Saggo", "Maximo Blanco", "Kiyoshi Tamura", "Aleksander Emelianenko", "Adlan Amagov", "Wilson Gouveia", "Jake O'Brien", "Polyana Viana", "Gilbert Melendez", "Anthony Perosh", "Zabit Magomedsharipov", "Rafael Cavalcante", "Dan Miller", "Junior Dos Santos", "Joe Soto", "Oluwale Bamgbose", "Jorge Rivera", "Yves Edwards", "Zubaira Tukhugov", "Chris Horodecki", "Patrick Cote", "George Roop", "Leslie Smith", "Chris Cariaso", "Walt Harris", "Javier Vazquez", "Vernon White", "Yair Rodriguez", "Urijah Faber", "Amanda Nunes", "Conor McGregor", "Lyoto Machida", "Igor Pokrajac", "Viktor Pesta", "Doug Marshall", "Thales Leites", "David Teymur", "Lorenz Larkin", "Kamaru Usman", "CB Dollaway", "Julianna Pena", "John Alessio", "Matt Schnell", "DaMarques Johnson", "Edwin Figueroa", "Andre Ewell", "Matt Wiman", "Kelvin Gastelum", "Lyman Good", "Jim Miller", "Josh Neer", "Alexander Gustafsson", "Josh Koscheck", "Oleg Taktarov", "Marlon Moraes", "Frank Camacho", "Amar Suloev", "TJ Grant", "Ikuhisa Minowa", "Bryan Barberena", "Leon Edwards", "Jessica Eye", "Adriano Martins", "Mike Brown", "Olivier Aubin-Mercier", "Uriah Hall", "Phil Baroni", "Nate Marquardt", "Pablo Garza", "Curtis Blaydes", "Micah Miller", "Rick Glenn", "Fedor Emelianenko", "Sara McMann", "Sergei Kharitonov", "Kevin Burns", "Elias Theodorou", "Tarec Saffiedine", "Derrick Lewis", "Eliot Marshall", "Will Campuzano", "Danny Castillo", "Caio Magalhaes", "Eric Schafer", "Eric Spicely", "Cathal Pendred", "Kamal Shalorus", "Chael Sonnen", "Michal Oleksiejczuk", "Song Yadong", "Dustin Poirier", "Robbie Lawler", "Aleksandar Rakic", "Ken Stone", "Georges St-Pierre", "Josh Emmett", "Justin Salas", "Nick Osipczak", "Akira Shoji", "Robert Peralta", "Claudia Gadelha", "Mike Whitehead", "Sammy Morgan", "Kazuo Misaki", "Marco Beltran", "Jerry Bohlander", "Marcus Hicks", "Scott Smith", "Mike van Arsdale", "Aspen Ladd", "Laverne Clark", "Karo Parisyan", "Irene Aldana", "Murilo Rua", "Chuck Liddell", "Tatiana Suarez", "Ricardo Ramos", "Jason MacDonald", "Chase Beebe", "Jan Blachowicz", "Shane del Rosario", "Anthony Pettis", "Calvin Kattar", "Marcus Davis", "Johny Hendricks", "Jussier Formiga", "Mark Hunt", "Brandon Vera", "Roger Bowling", "Tecia Torres", "Sheymon Moraes", "Junior Assuncao", "Ryan Gracie", "Luigi Fioravanti", "Trevor Smith", "Sijara Eubanks", "Mike Rodriguez", "Rob Font", "Wilson Reis", "Mitch Clarke", "Gegard Mousasi", "Rob Emerson", "Mike de la Torre", "Mike Swick", "Chris Gutierrez", "Jon Tuck", "Santiago Ponzinibbio", "Cody Stamann", "Masakazu Imanari", "Philip De Fries", "Alexa Grasso", "Bobby Green", "Corey Anderson", "Eddie Sanchez", "Joaquim Silva", "Claudio Silva", "Scott Askham", "Blagoy Ivanov", "Marcos Rogerio de Lima", "Jessica Aguilar", "Houston Alexander", "Mike Easton", "Cory Sandhagen", "Brian Kelleher", "Ken Shamrock", "Ian Heinisch", "Carmelo Marrero", "Paul Sass", "Randy Brown", "Karolina Kowalkiewicz", "Erick Silva", "Chan Sung Jung", "Rich Clementi", "Melvin Guillard", "Nam Phan", "Rafael Dos Anjos", "Drew McFedries", "Gian Villante", "Augusto Sakai", "Stephan Bonnar", "Shamil Abdurakhimov", "James Head", "Anthony Figueroa", "Mitsuhiro Ishida", "Erik Koch", "Kurt Holobaugh", "Kajan Johnson", "Kazuyuki Miyata", "Daniel Cormier", "Stevie Ray", "George Sullivan", "Nobuhiko Takada", "Sanae Kikuta", "Duane Ludwig", "Tom Erikson", "Thiago Alves", "Louis Gaudinot", "Cat Zingano", "Hector Lombard", "Raphael Assuncao", "Joe Duffy", "Brandon Davis", "Brock Lesnar", "Michihiro Omigawa", "Caros Fodor", "Dominick Cruz", "Antonio Schembri", "Frank Trigg", "Blas Avena", "Michael Chiesa", "Montel Jackson", "Paulo Costa", "Antonio Silva", "Joshua Burkman", "Katsuyori Shibata", "Matt Ricehouse", "Diego Ferreira", "Gesias Cavalcante", "Court McGee", "John Moraga", "Fredson Paixao", "Mizuto Hirota", "Todd Moore", "Vitor Ribeiro", "Ion Cutelaba", "John Lineker", "Andrei Arlovski", "Julian Erosa", "Mackens Semerzier", "Ketlen Vieira", "Kiichi Kunimoto", "David Abbott", "Devin Clark", "Amir Sadollah", "Carlos Condit", "Johnny Eduardo", "Paul Craig", "Mike Perry", "Keith Wisniewski", "Khalil Rountree Jr.", "Dong Hyun Ma", "Joe Riggs", "Omari Akhmedov", "Vinny Magalhaes", "Fabricio Camoes", "Jeremy Horn", "Jonathan Goulet", "Brandon Moreno", "Eugene Jackson", "Luke Barnatt", "Justin Willis", "Naoyuki Kotani", "Terry Martin", "Chris Camozzi", "Yan Cabral", "Hyun Gyu Lim", "James Moontasri", "Thiago Moises", "Marlon Vera", "Ben Rothwell", "Rogerio Nogueira", "Semmy Schilt", "Seth Petruzelli", "Yves Jabouin", "Gerald Meerschaert", "Beneil Dariush", "David Mitchell", "Anthony Hamilton", "Dominique Steele", "Ivan Menjivar", "Randa Markos", "Sam Hoger", "Miguel Torres", "Rodrigo Gracie", "Enrique Barzola", "Maryna Moroz", "Pete Sell", "Sean O'Connell", "Akira Corassani", "Nick Catone", "Henry Miller", "Gillian Robertson", "Makwan Amirkhani", "Andrea Lee", "Kenny Robertson", "Travis Browne", "Manvel Gamburyan", "Cody McKenzie", "Jorge Masvidal", "Roy Nelson", "Ryan Bader", "Kenny Florian", "Justine Kish", "Reza Madadi", "Davey Grant", "Cub Swanson", "Damacio Page", "Yoel Romero", "James Terry", "Matt Serra", "John Maguire", "Ricardo Almeida", "Lando Vannata", "John Makdessi", "Gleison Tibau", "Rory MacDonald", "Darren Stewart", "Leonardo Santos", "Luke Rockhold", "Evan Dunham", "Paul Kelly", "Gina Carano", "Josh Grispi", "Edson Barboza", "Max Holloway", "Dan Hardy", "Lucie Pudilova", "Antoni Hardonk", "Darren Till", "Mark Hall", "BJ Penn", "Falaniko Vitale", "Chris Brennan", "Lucas Martins", "Igor Vovchanchyn", "Zak Ottow", "Yancy Medeiros", "Takanori Gomi", "Constantinos Philippou", "Damien Brown", "Leandro Issa", "Jalin Turner", "Brian Ortega", "Shamar Bailey", "Yana Kunitskaya", "Dhiego Lima", "Lauren Murphy", "Alexander Hernandez", "Kenichi Yamamoto", "Brian Johnston", "Nate Diaz", "Trevin Giles", "Paul Felder", "Davi Ramos", "Forrest Griffin", "Hideo Tokoro", "Mirko Filipovic", "Molly McCann", "Leonard Garcia", "Rob McCullough", "Ji Yeon Kim", "Godofredo Pepey", "Jason Miller", "Tomasz Drwal", "Tim Hague", "Tyson Pedro", "Benji Radach", "Edmen Shahbazyan", "Ashlee Evans-Smith", "Forrest Petz", "Joby Sanchez", "Hidehiko Yoshida", "Dan Ige", "Jason High", "Oskar Piechota", "Sarah Moras", "Clay Guida", "Tim Credeur", "Julio Arce", "Kevin Holland", "Quinton Jackson", "Tae Hyun Bang", "Rafaello Oliveira", "Tim Boetsch", "Jingliang Li", "Cortney Casey", "Daniel Kelly", "Tony Ferguson", "Justin Wilcox", "Merab Dvalishvili", "Billy Evangelista", "Drakkar Klose", "Jared Rosholt", "Sean Spencer", "Jonathan Brookins", "Paul Varelans", "Eryk Anders", "Pete Spratt", "Francis Ngannou", "Paul Taylor", "Aaron Riley", "Chris Clements", "Kailin Curran", "Felipe Arantes", "Cezar Ferreira", "Russell Doane", "Wesley Correira", "Yuki Kondo", "Tom Watson", "Marius Zaromskis", "Tim Elliott", "Misha Cirkunov", "Eric Shelton", "Vaughan Lee", "Johnny Walker", "Lina Lansberg", "Anthony Ruiz"]