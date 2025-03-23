<html>
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>Door Access</title>
	<link rel="stylesheet" href="fontawesome/css/all.min.css"> <!-- https://fontawesome.com/ -->
	<link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro&display=swap" rel="stylesheet"> <!-- https://fonts.google.com/ -->
    <link href="css/bootstrap.min.css" rel="stylesheet">
    <link href="css/templatemo-xtra-blog.css" rel="stylesheet">
<!--
    
TemplateMo 553 Xtra Blog

https://templatemo.com/tm-553-xtra-blog

-->
</head>
<body>
	<header class="tm-header" id="tm-header">
        <div class="tm-header-wrapper">
            <button class="navbar-toggler" type="button" aria-label="Toggle navigation">
                <i class="fas fa-bars"></i>
            </button>
            <div class="tm-site-header">
                <div class="mb-3 mx-auto tm-site-logo"><i class="fas fa-times fa-2x"></i></div>            
                <h1 class="text-center">Door Access</h1>
            </div>
            <nav class="tm-nav" id="tm-nav">            
                <ul>
                    <li class="tm-nav-item"><a href="" class="tm-nav-link">
                        <i class="fas fa-home"></i>
                        Home
                    </a></li>
                  
                </ul>
            </nav>
            <div class="tm-mb-65">
                <a href="https://facebook.com" class="tm-social-link">
                    <i class="fab fa-facebook tm-social-icon"></i>
                </a>
                <a href="https://twitter.com" class="tm-social-link">
                    <i class="fab fa-twitter tm-social-icon"></i>
                </a>
                <a href="https://instagram.com" class="tm-social-link">
                    <i class="fab fa-instagram tm-social-icon"></i>
                </a>
                <a href="https://linkedin.com" class="tm-social-link">
                    <i class="fab fa-linkedin tm-social-icon"></i>
                </a>
            </div>
            <p class="tm-mb-80 pr-5 text-white">
                Xtra Blog is a multi-purpose HTML template from TemplateMo website. Left side is a sticky menu bar. Right side content will scroll up and down.
            </p>
        </div>
    </header>
    <div class="container-fluid">
        <main class="tm-main">
            <!-- Search form -->
            <div class="row tm-row">
                  
            </div>            
            <div class="row tm-row">
                
            </div>
            <div class="row tm-row">
                <div class="col-lg-8 tm-post-col">
                    <div class="tm-post-full">                    
                        
                         <?php
extract($_REQUEST);

$ff=fopen("pin.txt","r");
$pinn=fread($ff,filesize("pin.txt"));

if($act=="")
{
$f2=fopen("log.txt","w");
$val="1-1-1-1-1";
fwrite($f2,$val);
}
/*if($a=="1")
{
$f2=fopen("log.txt","w");
fwrite($f2,"1");
$msg="Accepted";
}
else if($a=="1")
{
$f2=fopen("log.txt","w");
fwrite($f2,"2");
$msg="Rejected";
}
else
{
$f2=fopen("log.txt","w");
fwrite($f2,"3");
$msg="";
}*/
if($act=="2")
{

$f2=fopen("log.txt","w");
$val="4-close-1-1-1";
fwrite($f2,$val);
?>
<script language="javascript">
window.location.href="ck.php?id=<?php echo $id; ?>&act=ok";
</script>
<?php							
}
if(isset($btn2))
{

$f2=fopen("log.txt","w");
$val="3-save-$name-$mobile-$detail";
fwrite($f2,$val);
?>
<script language="javascript">
window.location.href="ck.php?id=<?php echo $id; ?>&act=success";
</script>
<?php								
}
if($act=="4")
{

$f2=fopen("log.txt","w");
$val="4-close-1-1-1";
fwrite($f2,$val);
?>
<script language="javascript">
window.location.href="ck.php?id=<?php echo $id; ?>&act=ok";
</script>
<?php							
}
?>
                        <!-- Comments -->
                        <div>
                            <h2 class="tm-color-primary tm-post-title">Verification</h2>
                           
                            <p align="center"><img src="upload/img<?php echo $id; ?>.png" width="150" height="150" /></p>
							
							<p>
							<a href="ck.php?act=1&id=<?php echo $id; ?>" class="tm-btn tm-btn-primary tm-btn-small">Open</a> /
							<a href="ck.php?act=2&id=<?php echo $id; ?>" class="tm-btn tm-btn-primary tm-btn-small">Block</a>
							</p>
                            
                            <form method="post" name="form1" action="" class="mb-5 tm-comment-form">
                                <?php
								if($act=="1")
								{
								?>
                                <div class="mb-4">
                                    <input class="form-control" name="pin" type="text" placeholder="Enter the PIN">
                                </div>
                               
                                <div class="text-right">
                                    <input type="submit" name="btn" class="tm-btn tm-btn-primary tm-btn-small" value="Submit">                        
                                </div>   
								<?php
								}
								
								if(isset($btn))
								{
								if($pin==$pinn)
								{
									$f2=fopen("log.txt","w");
									$val="2-open-1-1-1";
									fwrite($f2,$val);
									$msg="Accepted";

								?>
								<h4>Are you save this person details for future purpose?</h4>
								<p>
							<a href="ck.php?act=3&id=<?php echo $id; ?>" class="tm-btn tm-btn-primary tm-btn-small">Yes</a> /
							<a href="ck.php?act=4&id=<?php echo $id; ?>" class="tm-btn tm-btn-primary tm-btn-small">No</a>
							</p>
								<?php
								
								}
								else
								{
								?><h3 style="color:#FF0000">PIN Wrong!</h3><?php
								}
								}
								
								
								if($act=="3")
								{
								?>
								<div class="mb-4">
                                    <input class="form-control" name="name" type="text" placeholder="Enter the Name">
                                </div>
								<div class="mb-4">
                                    <input class="form-control" name="mobile" type="text" placeholder="Enter the Mobile No.">
                                </div>
								<div class="mb-4">
                                    <input class="form-control" name="detail" type="text" placeholder="Person Detail">
                                </div>
								<div class="text-right">
                                    <input type="submit" name="btn2" class="tm-btn tm-btn-primary tm-btn-small" value="Save">                        
                                </div> 
								<p>&nbsp;</p>
								
								<?php
								}
								
								if($act=="success")
								{
								?><h3>Data Stored..</h3><?php
								}
								?>
								
								                             
                            </form>        
							
							
							                    
                        </div>
                    </div>
                </div>
                <aside class="col-lg-4 tm-aside-col">
                    <div class="tm-post-sidebar">
                        <hr class="mb-3 tm-hr-primary">
                        <h2 class="mb-4 tm-post-title tm-color-primary">Categories</h2>
                        <ul class="tm-mb-75 pl-5 tm-category-list">
                            <li><a href="#" class="tm-color-primary">Visual Designs</a></li>
                            <li><a href="#" class="tm-color-primary">Travel Events</a></li>
                            <li><a href="#" class="tm-color-primary">Web Development</a></li>
                            <li><a href="#" class="tm-color-primary">Video and Audio</a></li>
                            <li><a href="#" class="tm-color-primary">Etiam auctor ac arcu</a></li>
                            <li><a href="#" class="tm-color-primary">Sed im justo diam</a></li>
                        </ul>
                        <hr class="mb-3 tm-hr-primary">
                        <h2 class="tm-mb-40 tm-post-title tm-color-primary">Related Posts</h2>
                        <a href="#" class="d-block tm-mb-40">
                            <figure>
                                <img src="img/img-02.jpg" alt="Image" class="mb-3 img-fluid">
                                <figcaption class="tm-color-primary">Duis mollis diam nec ex viverra scelerisque a sit</figcaption>
                            </figure>
                        </a>
                        <a href="#" class="d-block tm-mb-40">
                            <figure>
                                <img src="img/img-05.jpg" alt="Image" class="mb-3 img-fluid">
                                <figcaption class="tm-color-primary">Integer quis lectus eget justo ullamcorper ullamcorper</figcaption>
                            </figure>
                        </a>
                        <a href="#" class="d-block tm-mb-40">
                            <figure>
                                <img src="img/img-06.jpg" alt="Image" class="mb-3 img-fluid">
                                <figcaption class="tm-color-primary">Nam lobortis nunc sed faucibus commodo</figcaption>
                            </figure>
                        </a>
                    </div>                    
                </aside>
            </div>
            <footer class="row tm-row">
                <div class="col-md-6 col-12 tm-color-gray">
                     <a rel="nofollow" target="_parent" href="https://templatemo.com" class="tm-external-link"></a>
                </div>
                <div class="col-md-6 col-12 tm-color-gray tm-copyright">
                   Door Access
                </div>
            </footer>
        </main>
    </div>
    <script src="js/jquery.min.js"></script>
    <script src="js/templatemo-script.js"></script>
</body>
</html>