import numpy as np
import matpotlib.pyplot as plt

;; .emacs                                                                                                               

;;; uncomment this line to disable loading of "default.el" at startup                                                   
;; (setq inhibit-default-init t)                                                                                        

;; enable visual feedback on selections                                                                                 
;(setq transient-mark-mode t)                                                                                           

;; default to better frame titles                                                                                       

;; Added by Package.el.  This must come before configurations of                                                        
;; installed packages.  Don't delete this line.  If you don't want it,                                                  
;; just comment it out by adding a semicolon to the start of the line.                                                  
;; You may delete these explanatory comments.                                                                           
(package-initialize)

(setq frame-title-format
      (concat  "%b - emacs@" (system-name)))

;; default to unified diffs                                                                                             
(setq diff-switches "-u")

;; always end a file with a newline                                                                                     
;(setq require-final-newline 'query)                                                                                    

;;; uncomment for CJK utf-8 support for non-Asian users                                                                 
;; (require 'un-define)                                                                                                 

(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(package-selected-packages (quote (elpy))))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 )

;; Installing elpy:
(package-initialize)
(elpy-enable)


;; Disable Elpy indentation highlight
(add-hook 'python-mode-hook (highlight-indentation-mode 0))

;; Custom key-bindings                                                         
(global-set-key (kbd "<f5>") (kbd "C-u C-c C-c"))
(global-set-key (kbd "<f6>") (kbd "C-c C-c"))
(global-set-key "\C-x\C-b" 'buffer-menu)

;; Add-ons
(ido-mode t)
(setq elpy-rpc-backend "jedi")
